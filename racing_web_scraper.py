#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: racing_web_scraper.py
# Description: 競馬情報のインターネット検索と取得モジュール

import re
import os
import time
import json
import random
import logging
import requests
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta, date
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, quote
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# User-Agentのリスト (検出回避のためランダム使用)
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:94.0) Gecko/20100101 Firefox/94.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36'
]

# リクエスト間の遅延時間設定 (秒)
MIN_REQUEST_DELAY = 1.0
MAX_REQUEST_DELAY = 3.0

# キャッシュ関連設定
DEFAULT_CACHE_DIR = './cache/web'
DEFAULT_CACHE_EXPIRY = 24 * 60 * 60  # 24時間 (秒)

class RacingWebScraper:
    """競馬情報を各種Webサイトから取得するクラス"""
    
    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR, logger: Optional[logging.Logger] = None,
                 min_delay: float = MIN_REQUEST_DELAY, max_delay: float = MAX_REQUEST_DELAY,
                 cache_expiry: int = DEFAULT_CACHE_EXPIRY, use_cache: bool = True):
        """
        初期化
        
        Args:
            cache_dir: キャッシュディレクトリのパス
            logger: ロガーインスタンス
            min_delay: リクエスト間の最小遅延時間 (秒)
            max_delay: リクエスト間の最大遅延時間 (秒)
            cache_expiry: キャッシュの有効期限 (秒)
            use_cache: キャッシュを使用するかどうか
        """
        self.logger = logger or logging.getLogger('racing_web_scraper')
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.cache_dir = Path(cache_dir)
        self.cache_expiry = cache_expiry
        self.use_cache = use_cache
        self.session = requests.Session()
        self.last_request_time = 0
        
        # キャッシュディレクトリの作成
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            for subdir in ['jockey', 'horse', 'race', 'odds', 'track']:
                (self.cache_dir / subdir).mkdir(exist_ok=True)
        
        self.logger.info(f"RacingWebScraper initialized - Cache: {self.use_cache}, Delay: {min_delay}-{max_delay}s")

    def _get_cache_path(self, category: str, key: str) -> Path:
        """カテゴリとキーからキャッシュファイルのパスを取得"""
        safe_key = re.sub(r'[^\w\-.]', '_', key)
        return self.cache_dir / category / f"{safe_key}.json"
    
    def _save_to_cache(self, category: str, key: str, data: Any) -> bool:
        """データをキャッシュに保存"""
        if not self.use_cache:
            return False
            
        try:
            cache_path = self._get_cache_path(category, key)
            cache_data = {
                'timestamp': datetime.now().timestamp(),
                'data': data
            }
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, default=str, indent=2)
            return True
        except Exception as e:
            self.logger.warning(f"Failed to save cache for {category}/{key}: {e}")
            return False
    
    def _load_from_cache(self, category: str, key: str) -> Optional[Any]:
        """キャッシュからデータを読み込む"""
        if not self.use_cache:
            return None
            
        cache_path = self._get_cache_path(category, key)
        if not cache_path.exists():
            return None
            
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                
            timestamp = cache_data.get('timestamp', 0)
            now = datetime.now().timestamp()
            
            # キャッシュ有効期限をチェック
            if now - timestamp > self.cache_expiry:
                self.logger.debug(f"Cache expired for {category}/{key}")
                return None
                
            return cache_data.get('data')
        except Exception as e:
            self.logger.warning(f"Failed to load cache for {category}/{key}: {e}")
            return None

    def _make_request(self, url: str, method: str = 'GET', params: Optional[Dict] = None, 
                     headers: Optional[Dict] = None, data: Optional[Dict] = None) -> Optional[requests.Response]:
        """リクエストを実行 (遅延とUser-Agentのランダム化)"""
        # 前回のリクエストからの経過時間を計算し、必要に応じて待機
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
            
        # ランダムな追加遅延
        if self.max_delay > self.min_delay:
            time.sleep(random.uniform(0, self.max_delay - self.min_delay))
        
        # デフォルトのヘッダー設定
        _headers = {
            'User-Agent': random.choice(USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.7,en;q=0.3',
        }
        
        if headers:
            _headers.update(headers)
        
        try:
            self.last_request_time = time.time()
            response = self.session.request(method, url, params=params, headers=_headers, data=data, timeout=30)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            self.logger.error(f"Request failed for {url}: {e}")
            return None

    # ------ 競走馬情報取得 ------ #
    def get_horse_info(self, horse_name: str, force_refresh: bool = False) -> Optional[Dict]:
        """競走馬の情報を取得"""
        category = 'horse'
        # キャッシュをチェック
        if not force_refresh:
            if cached := self._load_from_cache(category, horse_name):
                self.logger.debug(f"Using cached data for horse: {horse_name}")
                return cached
        
        # 実際の馬情報取得処理をここに実装
        horse_info = {
            'name': horse_name,
            'profile_data': {},
            'race_history': [],
            'last_updated': datetime.now().isoformat()
        }
        
        # キャッシュに保存
        self._save_to_cache(category, horse_name, horse_info)
        return horse_info
    
    def get_jockey_form(self, jockey_name: str, days: int = 30, force_refresh: bool = False) -> Optional[Dict]:
        """騎手の最近の調子を取得"""
        category = 'jockey'
        cache_key = f"{jockey_name}_{days}days"
        
        # キャッシュをチェック
        if not force_refresh:
            if cached := self._load_from_cache(category, cache_key):
                self.logger.debug(f"Using cached data for jockey: {jockey_name} (past {days} days)")
                return cached
        
        # 騎手情報の基本構造
        jockey_form = {
            'name': jockey_name,
            'recent_stats': {
                'total_races': 0,
                'wins': 0,
                'win_rate': 0.0,
                'top3_rate': 0.0,
                'condition': 'normal'  # 'good', 'normal', 'poor'
            },
            'race_results': [],
            'analysis': {},
            'last_updated': datetime.now().isoformat()
        }
        
        # キャッシュに保存
        self._save_to_cache(category, cache_key, jockey_form)
        return jockey_form
    
    def get_track_bias(self, track_name: str, race_date: Optional[Union[str, date]] = None, 
                      force_refresh: bool = False) -> Optional[Dict]:
        """トラックのバイアス情報を取得"""
        category = 'track'
        date_str = race_date.strftime('%Y%m%d') if isinstance(race_date, date) else \
                  (race_date if isinstance(race_date, str) else 'latest')
        cache_key = f"{track_name}_{date_str}"
        
        # キャッシュをチェック
        if not force_refresh:
            if cached := self._load_from_cache(category, cache_key):
                self.logger.debug(f"Using cached data for track bias: {track_name} ({date_str})")
                return cached
                
        # 仮のバイアスデータ
        track_bias = {
            'track': track_name,
            'date': date_str if date_str != 'latest' else datetime.now().strftime('%Y%m%d'),
            'surface': {
                'turf': {
                    'position': 'outside',
                    'running_style': 'front',
                    'confidence': 0.7,
                },
                'dirt': {
                    'position': 'inside',
                    'running_style': 'closer',
                    'confidence': 0.6,
                }
            },
            'weather': 'sunny',
            'track_condition': {
                'turf': '良',
                'dirt': '稍重'
            },
            'last_updated': datetime.now().isoformat()
        }
        
        # キャッシュに保存
        self._save_to_cache(category, cache_key, track_bias)
        return track_bias
    
    def get_race_entries(self, race_id: str, force_refresh: bool = False) -> Optional[Dict]:
        """レースのエントリー情報を取得"""
        category = 'race'
        
        # キャッシュをチェック
        if not force_refresh:
            if cached := self._load_from_cache(category, race_id):
                self.logger.debug(f"Using cached data for race: {race_id}")
                return cached
        
        # レース情報の基本構造
        race_info = {
            'race_id': race_id,
            'race_name': '',
            'track': '',
            'course': '',
            'distance': 0,
            'date': '',
            'start_time': '',
            'race_number': 0,
            'entries': [],
            'last_updated': datetime.now().isoformat()
        }
        
        # キャッシュに保存
        self._save_to_cache(category, race_id, race_info)
        return race_info
    
    def get_odds_data(self, race_id: str, odds_type: str = 'win', force_refresh: bool = False) -> Optional[Dict]:
        """レースのオッズ情報を取得"""
        category = 'odds'
        cache_key = f"{race_id}_{odds_type}"
        
        # キャッシュをチェック
        if not force_refresh:
            if cached := self._load_from_cache(category, cache_key):
                self.logger.debug(f"Using cached data for odds: {race_id} ({odds_type})")
                return cached
        
        odds_data = {
            'race_id': race_id,
            'odds_type': odds_type,
            'odds': {},
            'last_updated': datetime.now().isoformat()
        }
        
        # キャッシュに保存
        self._save_to_cache(category, cache_key, odds_data)
        return odds_data
    
    def get_horse_training(self, horse_name: str, days: int = 14, force_refresh: bool = False) -> Optional[Dict]:
        """馬の調教情報を取得"""
        category = 'horse'
        cache_key = f"{horse_name}_training_{days}days"
        
        # キャッシュをチェック
        if not force_refresh:
            if cached := self._load_from_cache(category, cache_key):
                self.logger.debug(f"Using cached data for horse training: {horse_name}")
                return cached
        
        training_data = {
            'horse_name': horse_name,
            'training_records': [],
            'analysis': {
                'condition': 'normal',
                'trend': 'stable',
                'note': ''
            },
            'last_updated': datetime.now().isoformat()
        }
        
        # キャッシュに保存
        self._save_to_cache(category, cache_key, training_data)
        return training_data

    def get_multiple_horse_info(self, horse_names: List[str], max_workers: int = 4) -> Dict[str, Dict]:
        """複数の馬情報を並列で取得"""
        results = {}
        
        def fetch_horse(name):
            return name, self.get_horse_info(name)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fetch_horse, name) for name in horse_names]
            for future in futures:
                name, data = future.result()
                if data:
                    results[name] = data
        
        return results

    def get_multiple_jockey_form(self, jockey_names: List[str], days: int = 30, max_workers: int = 4) -> Dict[str, Dict]:
        """複数の騎手情報を並列で取得"""
        results = {}
        
        def fetch_jockey(name):
            return name, self.get_jockey_form(name, days)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fetch_jockey, name) for name in jockey_names]
            for future in futures:
                name, data = future.result()
                if data:
                    results[name] = data
        
        return results

    def analyze_race_comprehensive(self, race_id: str, force_refresh: bool = False) -> Dict:
        """レースの包括的な分析を行う"""
        # レース情報を取得
        race_info = self.get_race_entries(race_id, force_refresh)
        if not race_info:
            return {'error': 'Failed to fetch race information'}
        
        # トラックのバイアス情報を取得
        track_bias = self.get_track_bias(race_info.get('track', ''), race_info.get('date', None), force_refresh)
        
        # 出走馬の名前リストを抽出
        horse_names = [entry.get('horse_name', '') for entry in race_info.get('entries', [])]
        
        # 騎手の名前リストを抽出
        jockey_names = [entry.get('jockey', '') for entry in race_info.get('entries', []) if entry.get('jockey')]
        
        # 馬情報を並列取得
        horse_info_dict = self.get_multiple_horse_info(horse_names)
        
        # 騎手情報を並列取得
        jockey_form_dict = self.get_multiple_jockey_form(jockey_names)
        
        # 調教情報を取得
        training_info_dict = {}
        for horse_name in horse_names:
            training_info = self.get_horse_training(horse_name, force_refresh=force_refresh)
            if training_info:
                training_info_dict[horse_name] = training_info
        
        # オッズ情報を取得
        odds_data = self.get_odds_data(race_id, 'win', force_refresh)
        
        # すべての情報を統合
        comprehensive_data = {
            'race_info': race_info,
            'track_bias': track_bias,
            'horse_info': horse_info_dict,
            'jockey_form': jockey_form_dict,
            'training_info': training_info_dict,
            'odds_data': odds_data,
            'analysis_time': datetime.now().isoformat()
        }
        
        return comprehensive_data

# 使用例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    scraper = RacingWebScraper()
    
    # 例: 馬情報の取得
    horse_info = scraper.get_horse_info("コントレイル")
    print(f"Horse Info: {horse_info['name'] if horse_info else 'Not found'}")
    
    # 例: レース情報の取得
    race_info = scraper.get_race_entries("202305021211")
    print(f"Race entries: {len(race_info['entries']) if race_info else 0}")
