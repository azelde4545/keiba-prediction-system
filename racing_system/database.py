#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
競馬予想システム - データベース管理モジュール

SQLiteデータベースを使用したデータの永続化と管理を行います。
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Union, Any
from contextlib import contextmanager

from .exceptions import DatabaseError, ValidationError
from .constants import Constants
from .utils import (
    parse_japanese_date,
    safe_int,
    safe_float,
    calculate_file_hash
)


class RacingDatabase:
    """競馬データベース管理クラス"""
    
    def __init__(self, db_path: Union[str, Path], logger_instance: logging.Logger):
        """
        初期化
        
        Args:
            db_path: データベースファイルのパス
            logger_instance: ロガーインスタンス
        """
        self.db_path = Path(db_path)
        self.logger = logger_instance
        self._initialize_database()
    
    @contextmanager
    def get_cursor(self) -> sqlite3.Cursor:
        """データベースカーソルを取得するコンテキストマネージャ"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode=WAL;")
            cursor = conn.cursor()
            yield cursor
            conn.commit()
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            self.logger.error(f"DB Error: {e}", exc_info=True)
            raise DatabaseError(f"Database operation failed: {e}") from e
        finally:
            if conn:
                conn.close()
    
    def _execute_sql(self, cursor: sqlite3.Cursor, sql: str, params: tuple = ()):
        """SQLを実行する"""
        try:
            cursor.execute(sql, params)
        except sqlite3.Error as e:
            self.logger.error(f"SQL execution error: {e} | SQL: {sql[:100]}...", exc_info=True)
            raise DatabaseError(f"SQL execution failed: {e}") from e
    
    def _initialize_database(self):
        """データベースを初期化する"""
        try:
            with self.get_cursor() as cursor:
                # テーブル定義
                tables = {
                    'races': '''(
                        race_id TEXT PRIMARY KEY,
                        race_name TEXT,
                        race_date TEXT,
                        track TEXT,
                        race_number INTEGER,
                        surface TEXT,
                        distance INTEGER,
                        weather TEXT,
                        track_condition TEXT,
                        grade TEXT,
                        race_details TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )''',
                    
                    'horses': '''(
                        horse_id TEXT PRIMARY KEY,
                        horse_name TEXT NOT NULL,
                        birth_year INTEGER,
                        gender TEXT,
                        coat_color TEXT,
                        trainer TEXT,
                        owner TEXT,
                        breeding_farm TEXT,
                        father TEXT,
                        mother TEXT,
                        mother_father TEXT,
                        horse_details TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )''',
                    
                    'jockeys': '''(
                        jockey_id TEXT PRIMARY KEY,
                        jockey_name TEXT NOT NULL,
                        birth_date TEXT,
                        jockey_details TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )''',
                    
                    'race_entries': '''(
                        entry_id TEXT PRIMARY KEY,
                        race_id TEXT NOT NULL,
                        horse_id TEXT NOT NULL,
                        jockey_id TEXT,
                        horse_number INTEGER,
                        frame_number INTEGER,
                        weight REAL,
                        odds REAL,
                        popularity INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (race_id) REFERENCES races (race_id) ON DELETE CASCADE,
                        FOREIGN KEY (horse_id) REFERENCES horses (horse_id),
                        FOREIGN KEY (jockey_id) REFERENCES jockeys (jockey_id)
                    )''',
                    
                    'race_results': '''(
                        result_id TEXT PRIMARY KEY,
                        race_id TEXT NOT NULL,
                        horse_id TEXT NOT NULL,
                        jockey_id TEXT,
                        horse_number INTEGER,
                        finish_position INTEGER,
                        finish_time TEXT,
                        finish_time_sec REAL,
                        margin TEXT,
                        corner_positions TEXT,
                        final_3f_time REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (race_id) REFERENCES races (race_id) ON DELETE CASCADE,
                        FOREIGN KEY (horse_id) REFERENCES horses (horse_id),
                        FOREIGN KEY (jockey_id) REFERENCES jockeys (jockey_id),
                        UNIQUE (race_id, horse_id)
                    )''',
                    
                    'predictions': '''(
                        prediction_id TEXT PRIMARY KEY,
                        race_id TEXT NOT NULL,
                        horse_id TEXT NOT NULL,
                        predicted_position INTEGER,
                        win_probability REAL,
                        predicted_time REAL,
                        prediction_details TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (race_id) REFERENCES races (race_id) ON DELETE CASCADE,
                        FOREIGN KEY (horse_id) REFERENCES horses (horse_id),
                        UNIQUE (race_id, horse_id)
                    )''',
                    
                    'pdf_cache': '''(
                        pdf_hash TEXT PRIMARY KEY,
                        pdf_path TEXT,
                        extracted_data TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )'''
                }
                
                # テーブル作成
                for name, schema in tables.items():
                    self._execute_sql(cursor, f"CREATE TABLE IF NOT EXISTS {name} {schema}")
                
                # インデックス作成
                indices = [
                    ("idx_races_date_track_num", "races", "(race_date, track, race_number)"),
                    ("idx_horses_name", "horses", "horse_name"),
                    ("idx_jockeys_name", "jockeys", "jockey_name"),
                    ("idx_race_entries_race_horse", "race_entries", "(race_id, horse_id)"),
                    ("idx_race_results_race_horse", "race_results", "(race_id, horse_id)"),
                    ("idx_predictions_race_horse", "predictions", "(race_id, horse_id)"),
                    ("idx_pdf_cache_hash", "pdf_cache", "pdf_hash")
                ]
                
                for idx, tbl, col in indices:
                    self._execute_sql(cursor, f"CREATE INDEX IF NOT EXISTS {idx} ON {tbl} {col}")
            
            self.logger.info("DB initialization/check complete.")
        except sqlite3.Error as e:
            self.logger.critical(f"DB init failed: {e}", exc_info=True)
            raise DatabaseError(f"Database initialization failed: {e}") from e
    
    def _generate_id(self, text: str, length: int = 16) -> str:
        """テキストからIDを生成する"""
        import hashlib
        import re
        
        if not (text and isinstance(text, str)):
            return f"invalid_{datetime.now().timestamp()}"
        
        normalized = re.sub(r'\s+', '', text).lower()
        return hashlib.sha1(normalized.encode('utf-8')).hexdigest()[:length]
    
    def _generate_race_id(self, info: Dict) -> Optional[str]:
        """レース情報からレースIDを生成する"""
        try:
            d_str = info.get('race_date')
            track = info.get('track')
            num = info.get('race_number')
            
            p_date = parse_japanese_date(d_str) if d_str else None
            
            if not (p_date and track and num is not None):
                self.logger.warning(
                    f"Cannot generate Race ID: Missing date({d_str}), track({track}), "
                    f"or number({num})"
                )
                return None
            
            track_code = Constants.TRACK_CODES.get(track, Constants.DEFAULT_TRACK_CODE)
            key = f"{p_date.strftime('%Y%m%d')}_{track_code}_{str(num).zfill(2)}_" \
                  f"{info.get('race_name','')}"
            
            return self._generate_id(key)
        except Exception as e:
            self.logger.error(f"Race ID generation error: {e}", exc_info=True)
            return None
    
    def _generate_horse_id(self, name: str) -> Optional[str]:
        """馬IDを生成する"""
        return self._generate_id(name) if name and isinstance(name, str) else None
    
    def _generate_jockey_id(self, name: str) -> Optional[str]:
        """騎手IDを生成する"""
        return self._generate_id(name) if name and isinstance(name, str) else None
    
    def store_race_info(self, race_info: Dict) -> Optional[str]:
        """レース情報を保存する"""
        if not isinstance(race_info, dict):
            raise ValidationError("Race info must be a dictionary")
        
        race_id = self._generate_race_id(race_info)
        if not race_id:
            raise ValidationError("Could not generate race ID from provided info")
        
        try:
            with self.get_cursor() as cursor:
                # レース日付のパース
                p_date = parse_japanese_date(race_info.get('race_date'))
                date_str = p_date.strftime('%Y-%m-%d') if p_date else race_info.get('race_date')
                
                # 詳細情報の準備
                excluded_keys = ['horses', 'validation_result', 'race_id']
                details = {k: v for k, v in race_info.items() if k not in excluded_keys}
                
                # レースデータの挿入
                race_data = (
                    race_id,
                    race_info.get('race_name'),
                    date_str,
                    race_info.get('track'),
                    safe_int(race_info.get('race_number')),
                    race_info.get('surface'),
                    safe_int(race_info.get('distance')),
                    race_info.get('weather'),
                    race_info.get('track_condition'),
                    race_info.get('grade'),
                    json.dumps(details, ensure_ascii=False, default=str)
                )
                
                self._execute_sql(
                    cursor,
                    """INSERT OR REPLACE INTO races 
                       (race_id, race_name, race_date, track, race_number, surface, 
                        distance, weather, track_condition, grade, race_details, created_at) 
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                    race_data
                )
            
            self.logger.info(f"Stored race {race_id} ('{race_info.get('race_name')}')")
            return race_id
            
        except sqlite3.Error as e:
            self.logger.error(f"DB error storing race {race_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to store race info: {e}") from e
    
    def get_race_info(self, race_id: str) -> Optional[Dict]:
        """レース情報を取得する"""
        if not race_id:
            return None
        
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT * FROM races WHERE race_id = ?", (race_id,))
                race_row = cursor.fetchone()
                
                if not race_row:
                    return None
                
                # レース情報の構築
                race_info = {**dict(race_row)}
                
                # race_detailsのJSONをマージ
                if race_row['race_details']:
                    try:
                        details = json.loads(race_row['race_details'])
                        race_info.update(details)
                    except json.JSONDecodeError:
                        pass
                
                if 'race_details' in race_info:
                    del race_info['race_details']
                
                race_info['horses'] = []
                return race_info
                
        except sqlite3.Error as e:
            self.logger.error(f"DB Error getting race {race_id}: {e}", exc_info=True)
            return None
    
    def get_horse_race_history(self, horse_id: str, limit: int = 10) -> List[Dict]:
        """馬のレース履歴を取得する"""
        if not horse_id:
            return []
        
        # 簡略化された実装
        return []
    
    def store_prediction(self, race_id: str, predictions: List[Dict]):
        """予測結果を保存する"""
        if not (race_id and predictions):
            return
        
        try:
            with self.get_cursor() as cursor:
                rows = []
                for p in predictions:
                    h_id = p.get('horse_id')
                    if not h_id:
                        continue
                    
                    details = {k: v for k, v in p.items() if k not in [
                        'scores', 'horse_id', 'predicted_rank', 'win_probability'
                    ]}
                    
                    rows.append((
                        f"pred_{race_id}_{h_id}",
                        race_id,
                        h_id,
                        safe_int(p.get('predicted_rank')),
                        safe_float(p.get('win_probability')),
                        safe_float(p.get('predicted_time')),
                        json.dumps(details, ensure_ascii=False, default=str)
                    ))
                
                if rows:
                    cursor.executemany(
                        """INSERT OR REPLACE INTO predictions 
                           (prediction_id, race_id, horse_id, predicted_position, 
                            win_probability, predicted_time, prediction_details, created_at) 
                           VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                        rows
                    )
                
                self.logger.info(f"Stored {len(rows)} predictions for race {race_id}.")
                
        except sqlite3.Error as e:
            self.logger.error(f"DB Error storing predictions for {race_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to store predictions: {e}") from e
    
    def cache_pdf_extraction(self, pdf_path: Union[str, Path], data: Dict):
        """PDF抽出結果をキャッシュする"""
        pdf_hash = calculate_file_hash(pdf_path)
        if not pdf_hash:
            return
        
        try:
            with self.get_cursor() as cursor:
                self._execute_sql(
                    cursor,
                    """INSERT OR REPLACE INTO pdf_cache 
                       (pdf_hash, pdf_path, extracted_data, created_at) 
                       VALUES (?, ?, ?, CURRENT_TIMESTAMP)""",
                    (pdf_hash, str(pdf_path), 
                     json.dumps(data, ensure_ascii=False, default=str))
                )
                
        except sqlite3.Error as e:
            self.logger.error(f"DB Error caching PDF {pdf_path}: {e}", exc_info=True)
    
    def get_cached_pdf_extraction(self, pdf_path: Union[str, Path]) -> Optional[Dict]:
        """キャッシュされたPDF抽出結果を取得する"""
        pdf_hash = calculate_file_hash(pdf_path)
        if not pdf_hash:
            return None
        
        try:
            with self.get_cursor() as cursor:
                cursor.execute(
                    "SELECT extracted_data FROM pdf_cache WHERE pdf_hash = ?",
                    (pdf_hash,)
                )
                row = cursor.fetchone()
                
                if row:
                    try:
                        return json.loads(row['extracted_data'])
                    except (json.JSONDecodeError, TypeError) as e:
                        self.logger.error(f"Cache decode error {pdf_path}: {e}")
                        return None
                
                return None
                
        except sqlite3.Error as e:
            self.logger.error(f"DB Error getting cache {pdf_path}: {e}", exc_info=True)
            return None
    
    def get_jockey_stats(self, jockey_id: str, days: int = 365) -> Dict:
        """騎手の統計情報を取得する"""
        stats = {
            'total_races': 0,
            'wins': 0,
            'seconds': 0,
            'thirds': 0,
            'win_ratio': 0.0,
            'place_ratio': 0.0,
            'top3_ratio': 0.0
        }
        
        # 簡略化された実装
        return stats
    
    def get_horse_info(self, horse_id: str) -> Optional[Dict]:
        """馬情報を取得する"""
        if not horse_id:
            return None
        
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT * FROM horses WHERE horse_id = ?", (horse_id,))
                row = cursor.fetchone()
                
                if row:
                    info = {**dict(row)}
                    if 'horse_details' in info:
                        del info['horse_details']
                    return info
                
                return None
                
        except sqlite3.Error as e:
            self.logger.error(f"DB Error getting horse {horse_id}: {e}", exc_info=True)
            return None