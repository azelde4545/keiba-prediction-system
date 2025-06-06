#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: keiba_predictor.py
# Description: 競馬予想システムのメインエントリーポイント

import os
import sys
import time
import argparse
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import yaml
import json

# モジュールインポート
from racing_system import (
    load_config, setup_logging, RacingDatabase, AdvancedPDFExtractor,
    HorseRacingAnalyzer, Constants, display_enhanced_analysis_results,
    display_detailed_pdf_extraction
)

# Web検索モジュールインポート (使用可能ならインポート)
try:
    from racing_web_scraper import RacingWebScraper
    WEB_SCRAPER_AVAILABLE = True
except ImportError:
    WEB_SCRAPER_AVAILABLE = False

# 設定ファイルパス
CONFIG_PATH = Path("./config/predictor_config.yaml")

class KeibaPredictor:
    """競馬予想システムの統合クラス"""
    
    def __init__(self, config_path: Path = CONFIG_PATH):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        # 設定の読み込み
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config['logging'])
        self.logger.info(f"競馬予想システム初期化中 - 設定: {config_path}")
        
        # データベース接続
        self.db = RacingDatabase(self.config['database']['path'], self.logger)
        
        # PDFパーサー初期化
        self.pdf_extractor = AdvancedPDFExtractor(self.db, self.config['pdf_extractor'], self.logger)
        
        # 分析エンジン初期化
        self.analyzer = HorseRacingAnalyzer(
            self.db, 
            self.config['analyzer'], 
            self.config['model_paths'], 
            self.logger, 
            self.config.get('odds_statistics')
        )
        
        # Web検索モジュール初期化 (可能な場合)
        self.web_scraper = None
        if WEB_SCRAPER_AVAILABLE:
            try:
                web_config = self.config.get('web_scraper', {})
                cache_dir = web_config.get('cache_dir', './cache/web')
                min_delay = web_config.get('min_delay', 1.0)
                max_delay = web_config.get('max_delay', 3.0)
                use_cache = web_config.get('use_cache', True)
                
                self.web_scraper = RacingWebScraper(
                    cache_dir=cache_dir,
                    logger=self.logger,
                    min_delay=min_delay,
                    max_delay=max_delay,
                    use_cache=use_cache
                )
                self.logger.info("Web検索モジュール初期化完了")
                
                # インターネット検索を許可するように設定を更新
                if not self.config['analyzer']['internet_features']['enabled']:
                    self.logger.info("インターネット検索機能を有効化します")
                    self.config['analyzer']['internet_features']['enabled'] = True
                    self.config['analyzer']['internet_features']['use_simulated_data'] = False
                    
                    # 分析エンジンに設定を反映
                    self.analyzer.enable_internet_features = True
                    self.analyzer.use_simulated_internet_data = False
                
            except Exception as e:
                self.logger.error(f"Web検索モジュールの初期化に失敗: {e}")
        else:
            self.logger.warning("Web検索モジュールが利用できません。racing_web_scraper.py がインポートできません。")
    
    def analyze_pdf(self, pdf_path: Union[str, Path], force_refresh: bool = False) -> Dict:
        """
        PDFファイルからレース情報を抽出して分析
        
        Args:
            pdf_path: PDFファイルのパス
            force_refresh: 既存のキャッシュを無視して強制的に抽出するか
            
        Returns:
            分析結果の辞書
        """
        self.logger.info(f"PDFファイル分析: {pdf_path}")
        
        # PDFからデータ抽出
        start_time = time.time()
        race_info = self.pdf_extractor.extract_race_info(pdf_path, force_refresh)
        extraction_time = time.time() - start_time
        
        if 'error' in race_info:
            self.logger.error(f"PDF抽出エラー: {race_info['error']}")
            return race_info
        
        # 抽出データを詳細表示
        display_detailed_pdf_extraction(race_info, self.config.get('formatting', {}))
        
        # Webからの追加情報取得
        if self.web_scraper:
            try:
                self._enhance_with_web_data(race_info)
            except Exception as e:
                self.logger.error(f"Web情報取得エラー: {e}")
        
        # レース分析実行
        start_time = time.time()
        analysis_result = self.analyzer.analyze_race(race_info)
        analysis_time = time.time() - start_time
        
        # 結果表示
        if 'error' not in analysis_result:
            self.logger.info(f"分析完了 - 抽出時間: {extraction_time:.2f}秒, 分析時間: {analysis_time:.2f}秒")
            display_enhanced_analysis_results(analysis_result)
            
            # 分析結果をデータベースに保存
            if 'analyzed_horses' in analysis_result:
                self.db.store_prediction(race_info.get('race_id'), analysis_result['analyzed_horses'])
                self.logger.info(f"予測結果をデータベースに保存しました ({len(analysis_result['analyzed_horses'])} 頭)")
        else:
            self.logger.error(f"分析エラー: {analysis_result['error']}")
        
        return analysis_result
    
    def analyze_race_id(self, race_id: str) -> Dict:
        """
        レースIDを指定して分析
        
        Args:
            race_id: レースID
            
        Returns:
            分析結果の辞書
        """
        self.logger.info(f"レースID分析: {race_id}")
        
        # データベースからレース情報取得
        race_info = self.db.get_race_info(race_id)
        if not race_info:
            # データベースにない場合はWeb検索
            if self.web_scraper:
                try:
                    self.logger.info(f"データベースにレース情報がありません。Webから検索します: {race_id}")
                    scraped_info = self.web_scraper.get_race_entries(race_id)
                    if scraped_info:
                        # Web情報をデータベース用に変換
                        race_info = self._convert_web_race_to_db_format(scraped_info)
                        # 変換したデータを保存
                        self.db.store_race_info(race_info)
                    else:
                        return {'error': f"レースID {race_id} の情報が見つかりません"}
                except Exception as e:
                    self.logger.error(f"Web検索エラー: {e}")
                    return {'error': f"レース情報取得エラー: {e}"}
            else:
                return {'error': f"レースID {race_id} の情報が見つかりません"}
        
        # Web情報で補完
        if self.web_scraper:
            try:
                self._enhance_with_web_data(race_info)
            except Exception as e:
                self.logger.error(f"Web情報取得エラー: {e}")
        
        # レース分析実行
        analysis_result = self.analyzer.analyze_race(race_info)
        
        # 結果表示
        if 'error' not in analysis_result:
            display_enhanced_analysis_results(analysis_result)
            
            # 分析結果をデータベースに保存
            if 'analyzed_horses' in analysis_result:
                self.db.store_prediction(race_id, analysis_result['analyzed_horses'])
        
        return analysis_result
    
    def _enhance_with_web_data(self, race_info: Dict) -> None:
        """
        Web情報で競馬データを強化
        
        Args:
            race_info: 強化するレース情報
        """
        if not self.web_scraper:
            return
        
        # 騎手情報の強化
        jockeys_to_process = []
        for horse in race_info.get('horses', []):
            jockey_name = horse.get('jockey')
            if jockey_name and jockey_name not in jockeys_to_process:
                jockeys_to_process.append(jockey_name)
        
        # 並列で騎手情報取得
        if jockeys_to_process:
            self.logger.info(f"騎手情報取得中: {len(jockeys_to_process)}名")
            jockey_data = self.web_scraper.get_multiple_jockey_form(jockeys_to_process)
            
            # 騎手情報を格納するためのフック関数を設定
            def get_jockey_web_data(data_type, key):
                if data_type == 'jockey_recent_form' and key in jockey_data:
                    jockey_info = jockey_data[key]
                    # 調子に応じたブースト/ペナルティ
                    condition = jockey_info.get('recent_stats', {}).get('condition', 'normal')
                    if condition == 'good':
                        return {'boost': jockey_info.get('analysis', {}).get('boost', 2.0)}
                    elif condition == 'poor':
                        return {'penalty': jockey_info.get('analysis', {}).get('penalty', 2.0)}
                return {}
            
            # 分析エンジンに騎手データ取得フックを設定
            self.analyzer._get_internet_data = get_jockey_web_data
        
        # トラックバイアスの取得
        track_name = race_info.get('track')
        race_date_str = race_info.get('race_date')
        if track_name and race_date_str:
            try:
                track_bias = self.web_scraper.get_track_bias(track_name, race_date_str)
                if track_bias:
                    self.logger.info(f"トラックバイアス情報取得: {track_name} {race_date_str}")
                    self.analyzer.race_track_bias_cache = track_bias
            except Exception as e:
                self.logger.error(f"トラックバイアス取得エラー: {e}")
    
    def _convert_web_race_to_db_format(self, web_race: Dict) -> Dict:
        """
        Web形式のレースデータをデータベース形式に変換
        
        Args:
            web_race: Web形式のレースデータ
            
        Returns:
            データベース形式のレースデータ
        """
        # 基本的なレース情報をコピー
        db_race = {
            'race_name': web_race.get('race_name', ''),
            'race_date': web_race.get('date', ''),
            'track': web_race.get('track', ''),
            'race_number': web_race.get('race_number', 0),
            'surface': web_race.get('course', ''),
            'distance': web_race.get('distance', 0),
            'start_time': web_race.get('start_time', ''),
            'horses': []
        }
        
        # 出走馬情報を変換
        for entry in web_race.get('entries', []):
            horse = {
                'horse_number': entry.get('horse_number', 0),
                'horse_name': entry.get('horse_name', ''),
                'frame_number': entry.get('frame_number', 0),
                'jockey': entry.get('jockey', ''),
                'weight': entry.get('weight', 0.0),
                'odds': entry.get('odds', 0.0),
                'popularity': entry.get('popularity', 0),
                'horse_weight': entry.get('horse_weight', 0),
                'weight_diff': entry.get('weight_diff', 0)
            }
            db_race['horses'].append(horse)
        
        # レースIDを生成
        race_id = self.db._generate_race_id(db_race)
        if race_id:
            db_race['race_id'] = race_id
        
        return db_race

def parse_arguments():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(description='競馬予想システム')
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--pdf', type=str, help='分析するPDFファイルのパス')
    input_group.add_argument('--race-id', type=str, help='分析するレースID')
    
    parser.add_argument('--force', action='store_true', help='キャッシュを無視して強制的に再解析')
    parser.add_argument('--no-web', action='store_true', help='Web検索を無効化')
    parser.add_argument('--config', type=str, help='使用する設定ファイルのパス', default=str(CONFIG_PATH))
    parser.add_argument('--detail', action='store_true', help='詳細なデバッグ情報を表示')
    
    return parser.parse_args()

def main():
    """メイン関数"""
    # コマンドライン引数の処理
    args = parse_arguments()
    
    # 設定ファイルパスの解決
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"設定ファイルが見つかりません: {config_path}")
        config_path = CONFIG_PATH
        
    # 詳細モードの設定
    if args.detail:
        # ログレベルを DEBUG に設定
        config = load_config(config_path)
        config['logging']['level'] = 'DEBUG'
        
        # 設定ファイルを一時的に更新
        temp_config_path = Path("./temp_config.yaml")
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        config_path = temp_config_path
    
    # 予測システムの初期化
    predictor = KeibaPredictor(config_path)
    
    # Web検索無効化オプション処理
    if args.no_web:
        predictor.web_scraper = None
        predictor.analyzer.enable_internet_features = False
        predictor.logger.info("Web検索機能を無効化しました")
    
    # 分析実行
    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"PDFファイルが見つかりません: {pdf_path}")
            return 1
        
        result = predictor.analyze_pdf(pdf_path, args.force)
    elif args.race_id:
        result = predictor.analyze_race_id(args.race_id)
    
    # 一時設定ファイルの削除
    if args.detail and config_path.name == "temp_config.yaml":
        try:
            config_path.unlink()
        except:
            pass
    
    # エラーがあれば終了コード1で終了
    if isinstance(result, dict) and 'error' in result:
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())