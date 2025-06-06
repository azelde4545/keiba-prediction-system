#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
競馬予想システム - PDF解析モジュール

PDFファイルからレース情報を抽出する機能を提供します。
"""

import re
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Union, Any, Tuple
from collections import defaultdict

import pandas as pd

from .exceptions import PDFExtractionError, ValidationError
from .constants import Constants
from .utils import (
    safe_int,
    safe_float,
    parse_japanese_date,
    calculate_file_hash
)
from .database import RacingDatabase

# PDFライブラリのインポート試行
PDF_LIBS_AVAILABLE = False
try:
    import PyPDF2
    from pdfminer.high_level import extract_text
    import tabula
    PDF_LIBS_AVAILABLE = True
except ImportError:
    pass


class AdvancedPDFExtractor:
    """高度なPDF抽出クラス"""
    
    def __init__(self, db: RacingDatabase, pdf_config: Dict, logger: logging.Logger):
        """
        初期化
        
        Args:
            db: データベースインスタンス
            pdf_config: PDF抽出設定
            logger: ロガー
        """
        self.db = db
        self.pdf_config = pdf_config
        self.logger = logger
        self.validation_config = pdf_config.get('validation', {})
        
        # パターンのコンパイル
        self.patterns = {
            name: re.compile(p) 
            for name, p in pdf_config.get('patterns', {}).items()
        }
        
        self.table_keywords = pdf_config.get('table_keywords', {
            'required': ['馬番', '馬名'],
            'optional': ['騎手', '斤量']
        })
        
        self.tabula_options = pdf_config.get('tabula_options', [
            {
                'lattice': True,
                'multiple_tables': True,
                'pages': 'all',
                'pandas_options': {'header': None}
            }
        ])
        
        if not isinstance(self.tabula_options, list) or not self.tabula_options:
            self.logger.warning("Invalid 'tabula_options' in config. Using default lattice.")
            self.tabula_options = [{
                'lattice': True,
                'multiple_tables': True,
                'pages': 'all',
                'pandas_options': {'header': None}
            }]
        
        if not PDF_LIBS_AVAILABLE:
            self.logger.warning("PDF libraries unavailable, extraction disabled.")
    
    def extract_race_info(self, pdf_path: Union[str, Path], 
                         force_refresh: bool = False) -> Dict:
        """
        PDFファイルからレース情報を抽出する
        
        Args:
            pdf_path: PDFファイルのパス
            force_refresh: キャッシュを無視して強制的に抽出するか
            
        Returns:
            レース情報辞書
        """
        pdf_p = Path(pdf_path)
        self.logger.info(f"Extracting PDF: {pdf_p} (Force: {force_refresh})")
        
        err_res = lambda msg: {
            'error': msg,
            'validation_result': {
                'result': 'ERROR',
                'details': [msg]
            }
        }
        
        if not PDF_LIBS_AVAILABLE:
            return err_res('PDF libs not installed.')
        
        if not pdf_p.is_file():
            return err_res(f'PDF not found: {pdf_p}')
        
        pdf_hash = calculate_file_hash(pdf_p)
        if not pdf_hash:
            return err_res('PDF hash failed.')
        
        # キャッシュチェック
        if not force_refresh and (cached := self.db.get_cached_pdf_extraction(pdf_p)):
            if 'validation_result' not in cached or \
               not isinstance(cached.get('validation_result'), dict):
                cached['validation_result'] = self.validate_extracted_data(cached)
            
            self.logger.info(f"Using cached data for {pdf_p}")
            
            # race_idの確認
            if 'race_id' not in cached:
                race_id_from_cache = self.db._generate_race_id(cached)
                if race_id_from_cache:
                    cached['race_id'] = race_id_from_cache
            
            return cached
        
        self.logger.info(f"No cache hit or force refresh for {pdf_p}. Extracting...")
        
        # PDFからテキスト抽出
        text = self._extract_text(pdf_p)
        
        # レース情報のパース
        race_info = self._parse_race_info_from_text(text)
        
        # 馬情報の抽出
        horses = self._extract_horses_from_pdf(pdf_p, race_info, text)
        
        # 結果の統合
        result = {**race_info, 'horses': horses}
        
        # 検証
        val_res = self.validate_extracted_data(result)
        result['validation_result'] = val_res
        
        # レースIDの生成
        race_id = self.db._generate_race_id(result)
        
        if race_id:
            result['race_id'] = race_id
            self.db.cache_pdf_extraction(pdf_p, result)
            self.logger.info(
                f"PDF extracted & cached (Validation: {val_res.get('result', 'UNK')}): "
                f"RaceID {race_id} ({result.get('race_name')})"
            )
        else:
            self.logger.error(
                f"RaceID generation failed for {pdf_p}. "
                f"Caching data, but analysis might be incomplete."
            )
            self.db.cache_pdf_extraction(pdf_p, result)
            result['error'] = result.get('error', 
                                       'Failed to generate race ID from extracted data.')
        
        return result
    
    def _extract_text(self, pdf_path: Path) -> str:
        """
        PDFからテキストを抽出する
        
        Args:
            pdf_path: PDFファイルパス
            
        Returns:
            抽出されたテキスト
        """
        text = ""
        
        try:
            use_pdfminer = self.pdf_config.get('use_pdfminer', True)
            
            if use_pdfminer and PDF_LIBS_AVAILABLE:
                try:
                    text = extract_text(pdf_path, caching=False)
                    if text:
                        self.logger.debug(f"Text extracted using pdfminer.six for {pdf_path}")
                except Exception as e_miner:
                    self.logger.warning(
                        f"pdfminer.six extraction failed for {pdf_path}: {e_miner}. "
                        f"Trying PyPDF2."
                    )
                    if PDF_LIBS_AVAILABLE:
                        text = self._extract_text_with_pypdf2(pdf_path)
            elif PDF_LIBS_AVAILABLE:
                text = self._extract_text_with_pypdf2(pdf_path)
            
            if not text:
                self.logger.warning(f"Could not extract text from PDF: {pdf_path}")
        except Exception as e_text:
            self.logger.error(f"Unexpected text extraction error {pdf_path}: {e_text}",
                            exc_info=True)
        
        return text
    
    def _extract_text_with_pypdf2(self, pdf_path: Path) -> str:
        """
        PyPDF2を使用してテキストを抽出する
        
        Args:
            pdf_path: PDFファイルパス
            
        Returns:
            抽出されたテキスト
        """
        text = ""
        
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                
                if reader.is_encrypted:
                    try:
                        if reader.decrypt('') == PyPDF2.PasswordType.NOT_DECRYPTED:
                            self.logger.warning(
                                f"PDF is encrypted and password decryption failed: {pdf_path}"
                            )
                            return ""
                    except Exception as decrypt_err:
                        self.logger.warning(
                            f"Error during decryption attempt for {pdf_path}: {decrypt_err}"
                        )
                        return ""
                
                page_texts = [p.extract_text() for p in reader.pages if p.extract_text()]
                text = "\n".join(page_texts)
            
            if text:
                self.logger.debug(f"Text extracted using PyPDF2 for {pdf_path}")
            else:
                self.logger.warning(f"PyPDF2 could not extract text from {pdf_path}")
        except Exception as e:
            self.logger.error(f"PyPDF2 extraction error {pdf_path}: {e}", exc_info=True)
        
        return text
    
    def _parse_race_info_from_text(self, text: str) -> Dict[str, Any]:
        """
        テキストからレース情報をパースする
        
        Args:
            text: 抽出されたテキスト
            
        Returns:
            レース情報辞書
        """
        info = defaultdict(lambda: None)
        
        if not text:
            return dict(info)
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        txt_norm = ' '.join(lines)
        
        # レース名とグレード
        if p := self.patterns.get('race_name'):
            if m := p.search(txt_norm):
                info['race_name'] = m.group(1).strip()
                info['grade'] = self._normalize_grade(m.group(2))
                
                if not info['grade'] and (p_alt := self.patterns.get('grade_alt')):
                    search_area = m.group(0) or txt_norm
                    if m_alt := p_alt.search(search_area):
                        info['grade'] = self._normalize_grade(m_alt.group(1))
        elif (p_alt := self.patterns.get('grade_alt')):
            if m_alt := p_alt.search(txt_norm):
                info['grade'] = self._normalize_grade(m_alt.group(1))
        
        # 日付
        if p := self.patterns.get('race_date'):
            if m := p.search(txt_norm):
                info['race_date'] = m.group(1).replace(' ', '')
        
        # トラックとレース番号
        if p := self.patterns.get('track_race'):
            if m := p.search(txt_norm):
                groups = m.groups()
                
                # トラック名を探す
                track = next(
                    (g for g in groups if g in Constants.TRACK_CODES),
                    None
                )
                if track:
                    info['track'] = track
                
                # レース番号を探す
                num = next(
                    (g for g in reversed(groups) 
                     if g and g.replace('R', '').isdigit()),
                    None
                )
                if num:
                    info['race_number'] = safe_int(num.replace('R', ''))
        
        # コースと距離
        if p := self.patterns.get('course'):
            if m := p.search(txt_norm):
                info['surface'] = m.group(1)
                info['distance'] = safe_int(m.group(2))
        
        # 天候と馬場状態
        if p := self.patterns.get('conditions'):
            if m := p.search(txt_norm):
                g = m.groups()
                weather_words = ['晴', '曇', '雨', '雪', '小雨', '小雪']
                cond_words = ['良', '稍', '重', '不']
                
                # 天候
                info['weather'] = next(
                    (w.strip() for w in [g[0], g[2]] 
                     if w and any(word in w for word in weather_words)),
                    None
                )
                
                # 馬場状態
                cond_str = next(
                    (c.strip() for c in [g[1], g[3]] 
                     if c and any(word in c for word in cond_words)),
                    None
                )
                
                if cond_str:
                    cond_norm = cond_str.replace('稍重', '稍').replace('不良', '不')
                    if '稍' in cond_norm:
                        info['track_condition'] = '稍重'
                    elif '不' in cond_norm:
                        info['track_condition'] = '不良'
                    elif '重' in cond_norm:
                        info['track_condition'] = '重'
                    elif '良' in cond_norm:
                        info['track_condition'] = '良'
        
        # 発走時刻
        if p := self.patterns.get('start_time'):
            if m := p.search(txt_norm):
                info['start_time'] = m.group(1).strip()
        
        return {k: v for k, v in info.items() if v is not None}
    
    def _normalize_grade(self, grade_str: Optional[str]) -> Optional[str]:
        """
        グレード文字列を正規化する
        
        Args:
            grade_str: グレード文字列
            
        Returns:
            正規化されたグレード
        """
        if not (grade_str and isinstance(grade_str, str)):
            return None
        
        g = re.sub(r'[()\[\]]', '', grade_str.upper().strip()
                  .replace('Ⅰ', '1').replace('Ⅱ', '2').replace('Ⅲ', '3'))
        
        if g in ['G1', 'G2', 'G3', 'L', 'OP']:
            return g
        elif 'LISTED' in g:
            return 'L'
        else:
            return None
    
    def _extract_horses_from_pdf(self, pdf_path: Path, race_info: Dict, 
                                text: str) -> List[Dict]:
        """
        PDFから馬情報を抽出する
        
        Args:
            pdf_path: PDFファイルパス
            race_info: レース情報
            text: 抽出されたテキスト
            
        Returns:
            馬情報のリスト
        """
        horses = []
        use_tabula = self.pdf_config.get('use_tabula', True)
        
        if not (use_tabula and PDF_LIBS_AVAILABLE):
            self.logger.info("Tabula extraction is disabled or unavailable.")
            return []
        
        for i, options in enumerate(self.tabula_options):
            self.logger.info(f"Attempting Tabula extraction with option set {i+1}: {options}")
            
            try:
                dfs = tabula.read_pdf(pdf_path, **options)
                
                if dfs:
                    self.logger.info(
                        f"Tabula (options {i+1}) found {len(dfs)} potential table(s)."
                    )
                    parsed_horses = self._parse_horses_from_tables(dfs, race_info)
                    
                    if parsed_horses:
                        self.logger.info(
                            f"Successfully parsed horses using Tabula options {i+1}."
                        )
                        horses = parsed_horses
                        break
                    else:
                        self.logger.info(
                            f"Parsing failed or yielded no horses with options {i+1}."
                        )
                else:
                    self.logger.info(f"Tabula (options {i+1}) found no tables.")
                    
            except Exception as e:
                self.logger.error(
                    f"Tabula extraction error with options {i+1} for {pdf_path}: {e}",
                    exc_info=True
                )
        
        if not horses:
            self.logger.warning(
                f"Could not extract horse table using any Tabula options from {pdf_path}."
            )
        
        return horses
    
    def _parse_horses_from_tables(self, tables: List[pd.DataFrame], 
                                 race_info: Dict) -> List[Dict]:
        """
        テーブルから馬情報をパースする
        
        Args:
            tables: DataFrameのリスト
            race_info: レース情報
            
        Returns:
            馬情報のリスト
        """
        potential = self._find_potential_horse_tables(tables)
        
        if not potential:
            self.logger.warning("No potential horse tables found in candidates.")
            return []
        
        df = potential[0]
        self.logger.info(f"Parsing horse data from table shape {df.shape}")
        
        # カラムマッピング
        header_map = self._map_columns(df)
        
        if not (header_map and 'horse_number' in header_map and 'horse_name' in header_map):
            self.logger.error(
                f"Failed map required cols (Num, Name) for best table candidate. "
                f"Map: {header_map}"
            )
            return []
        
        # データ開始行を見つける
        start_row = self._find_first_data_row(df, header_map.get('horse_number'))
        
        if start_row is None:
            self.logger.error("Could not find start data row in best table candidate.")
            return []
        
        self.logger.debug(f"Parsing data from row index: {start_row}")
        
        # 簡略化されたパーシング実装
        horses = []
        for idx in range(start_row, len(df)):
            try:
                row = df.iloc[idx]
                horse_num = safe_int(str(row.iloc[header_map['horse_number']]).strip())
                
                if not horse_num or horse_num <= 0:
                    continue
                
                horse_name = str(row.iloc[header_map['horse_name']]).strip()
                if not horse_name or horse_name in ['nan', 'NaN', '']:
                    continue
                
                horse_data = {
                    'horse_number': horse_num,
                    'horse_name': horse_name
                }
                
                # 騎手
                if 'jockey' in header_map:
                    jockey = str(row.iloc[header_map['jockey']]).strip()
                    if jockey and jockey not in ['nan', 'NaN']:
                        horse_data['jockey'] = jockey
                
                # 斤量
                if 'weight' in header_map:
                    weight = safe_float(str(row.iloc[header_map['weight']]).strip())
                    if weight and 40 <= weight <= 70:
                        horse_data['weight'] = weight
                
                horses.append(horse_data)
                
            except Exception as e:
                self.logger.debug(f"Error parsing row {idx}: {e}")
                continue
        
        self.logger.info(f"Successfully parsed {len(horses)} horses from table.")
        return horses
    
    def _find_potential_horse_tables(self, tables: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """馬テーブルの候補を見つける"""
        candidates = []
        
        for df in tables:
            if df.empty or df.shape[0] < 3:
                continue
            
            # テーブル内のテキストを結合
            all_text = ' '.join(str(cell) for row in df.values for cell in row if pd.notna(cell))
            
            # 必要キーワードのチェック
            required_keywords = self.table_keywords.get('required', [])
            if all(keyword in all_text for keyword in required_keywords):
                candidates.append(df)
        
        # サイズでソート（大きいテーブル優先）
        candidates.sort(key=lambda x: x.shape[0] * x.shape[1], reverse=True)
        
        return candidates
    
    def _map_columns(self, df: pd.DataFrame) -> Dict[str, int]:
        """カラムをマップする"""
        header_map = {}
        
        # 簡略化されたマッピング
        for col_idx in range(df.shape[1]):
            col_text = ' '.join(str(cell) for cell in df.iloc[:3, col_idx] if pd.notna(cell))
            
            if '馬番' in col_text or '番号' in col_text:
                header_map['horse_number'] = col_idx
            elif '馬名' in col_text:
                header_map['horse_name'] = col_idx
            elif '騎手' in col_text:
                header_map['jockey'] = col_idx
            elif '斤量' in col_text:
                header_map['weight'] = col_idx
        
        return header_map
    
    def _find_first_data_row(self, df: pd.DataFrame, horse_number_col: int) -> Optional[int]:
        """最初のデータ行を見つける"""
        for idx in range(len(df)):
            try:
                cell_value = str(df.iloc[idx, horse_number_col]).strip()
                if cell_value.isdigit() and int(cell_value) > 0:
                    return idx
            except (IndexError, ValueError):
                continue
        
        return None
    
    def validate_extracted_data(self, data: Dict) -> Dict:
        """
        抽出されたデータを検証する
        
        Args:
            data: 抽出データ
            
        Returns:
            検証結果
        """
        issues = []
        warnings = []
        
        # 基本情報のチェック
        if not data.get('race_name'):
            issues.append("レース名が見つかりません")
        
        if not data.get('track'):
            issues.append("トラック名が見つかりません")
        
        if not data.get('race_date'):
            warnings.append("レース日が見つかりません")
        
        # 馬情報のチェック
        horses = data.get('horses', [])
        if not horses:
            issues.append("出走馬情報が見つかりません")
        else:
            valid_horses = 0
            for horse in horses:
                if horse.get('horse_name') and horse.get('horse_number'):
                    valid_horses += 1
            
            if valid_horses == 0:
                issues.append("有効な馬情報がありません")
            elif valid_horses < 8:
                warnings.append(f"馬情報が少ないです: {valid_horses}頭")
        
        # 結果の判定
        if issues:
            result = 'FAILED'
        elif warnings:
            result = 'WARNING'
        else:
            result = 'SUCCESS'
        
        return {
            'result': result,
            'issues': issues,
            'warnings': warnings,
            'details': issues + warnings,
            'horse_count': len(horses) if horses else 0
        }