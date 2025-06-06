#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
競馬予想システム - メインモジュール

分離されたモジュールを統合し、後方互換性を提供します。
このファイルは既存のコードとの互換性のために残されています。
新規開発では、各モジュールを直接インポートすることを推奨します。
"""

# 基本的なインポート
import warnings

# 分離されたモジュールから全てをインポート
from racing_system.exceptions import *
from racing_system.constants import Constants
from racing_system.utils import *
from racing_system.database import RacingDatabase
from racing_system.pdf_parser import AdvancedPDFExtractor
from racing_system.analyzer import HorseRacingAnalyzer
from racing_system.display import (
    display_detailed_pdf_extraction,
    display_enhanced_analysis_results
)

# 型定義（必要に応じて）
from racing_system.types import *

# 後方互換性のための警告
warnings.warn(
    "racing_system.py is deprecated. "
    "Please import from specific modules in the racing_system package instead.",
    DeprecationWarning,
    stacklevel=2
)

# バージョン情報
__version__ = '3.2.0'
__all__ = [
    # 例外
    'RacingSystemError',
    'PDFExtractionError',
    'AnalysisError',
    'ValidationError',
    'DatabaseError',
    'ConfigError',
    'WebScrapingError',
    'ModelError',
    'SecurityError',
    
    # 定数
    'Constants',
    
    # ユーティリティ
    'merge_dicts',
    'load_config',
    'setup_logging',
    'get_roi_from_range_data',
    'get_statistical_win_rate',
    'calculate_odds_based_expected_value',
    'calculate_file_hash',
    'normalize_score',
    'parse_japanese_date',
    'format_finish_time',
    'parse_finish_time',
    'safe_float',
    'safe_int',
    'validate_file_path',
    
    # メインクラス
    'RacingDatabase',
    'AdvancedPDFExtractor',
    'HorseRacingAnalyzer',
    
    # 表示関数
    'display_detailed_pdf_extraction',
    'display_enhanced_analysis_results',
    
    # デフォルト設定（後方互換性）
    'DEFAULT_CONFIG',
    'DETAILED_ODDS_WIN_RATES',
]


# インターネット機能のヘルパー関数（後方互換性）
def _get_simulated_internet_data(data_type: str, key: str) -> Optional[Dict]:
    """
    インターネット検索結果をシミュレートするダミー関数
    
    後方互換性のために残されています。
    新規開発では使用しないでください。
    """
    import random
    logger = logging.getLogger('racing_system')
    logger.debug(f"Simulating internet search for {data_type}: {key}")
    
    if data_type == 'jockey_recent_form':
        r = random.random()
        if r < 0.1:
            return {'boost': random.uniform(2.0, 6.0)}
        elif r < 0.2:
            return {'penalty': random.uniform(2.0, 6.0)}
        else:
            return {}
    elif data_type == 'horse_latest_training':
        r = random.random()
        if r < 0.15:
            return {'boost': random.uniform(3.0, 7.0)}
        else:
            return {}
    elif data_type == 'track_bias':
        r = random.random()
        if r < 0.2:
            return {'bias': 'inner_front'}
        elif r < 0.4:
            return {'bias': 'outer_sashi'}
        else:
            return {'bias': 'flat'}
    
    return {}


def integrate_web_data(race_info: Dict, web_data: Dict) -> Dict:
    """
    Web検索から取得したデータをレース情報に統合する
    
    後方互換性のために残されています。
    新規開発では racing_web_scraper モジュールの機能を直接使用してください。
    
    Args:
        race_info: 既存のレース情報辞書
        web_data: Webから取得したデータ辞書
        
    Returns:
        統合されたレース情報
    """
    if not web_data or not isinstance(web_data, dict):
        return race_info
    
    result = race_info.copy()
    
    # 基本レース情報の更新
    base_race_info = web_data.get('race_info', {})
    if base_race_info:
        for key in ['race_name', 'track', 'race_number', 'surface', 'distance', 'start_time']:
            if not result.get(key) and base_race_info.get(key):
                result[key] = base_race_info[key]
    
    # トラックバイアス情報の追加
    track_bias = web_data.get('track_bias', {})
    if track_bias:
        result['track_bias'] = track_bias
    
    # 馬情報の補完・更新
    horse_info_dict = web_data.get('horse_info', {})
    jockey_form_dict = web_data.get('jockey_form', {})
    training_info_dict = web_data.get('training_info', {})
    
    if any([horse_info_dict, jockey_form_dict, training_info_dict]):
        for i, horse in enumerate(result.get('horses', [])):
            horse_name = horse.get('horse_name')
            jockey_name = horse.get('jockey')
            
            # 馬情報の追加
            if horse_name and horse_name in horse_info_dict:
                h_info = horse_info_dict[horse_name]
                if profile := h_info.get('profile_data', {}):
                    for key, value in profile.items():
                        if '父' in key and not horse.get('father'):
                            horse['father'] = value
                        elif '母' in key and not horse.get('mother'):
                            horse['mother'] = value
                        elif '生年月日' in key and not horse.get('birth_year'):
                            try:
                                import re
                                year_match = re.search(r'(\d{4})年', value)
                                if year_match:
                                    horse['birth_year'] = int(year_match.group(1))
                            except:
                                pass
                        elif '性別' in key and not horse.get('gender'):
                            horse['gender'] = value
            
            # 調教情報の追加
            if horse_name and horse_name in training_info_dict:
                training = training_info_dict[horse_name]
                horse['training_analysis'] = training.get('analysis', {})
                if records := training.get('training_records', []):
                    horse['recent_training'] = records[0] if records else {}
            
            # 騎手情報の追加
            if jockey_name and jockey_name in jockey_form_dict:
                j_form = jockey_form_dict[jockey_name]
                horse['jockey_form'] = j_form.get('recent_stats', {})
                horse['jockey_analysis'] = j_form.get('analysis', {})
    
    # オッズ情報の更新
    odds_data = web_data.get('odds_data', {})
    if odds_horses := odds_data.get('odds', {}):
        for i, horse in enumerate(result.get('horses', [])):
            h_num = horse.get('horse_number')
            if h_num and h_num in odds_horses:
                odds_info = odds_horses[h_num]
                if not horse.get('odds') and odds_info.get('odds'):
                    horse['odds'] = odds_info['odds']
                if not horse.get('popularity') and odds_info.get('popularity'):
                    horse['popularity'] = odds_info['popularity']
    
    # Web情報が統合されたフラグを追加
    result['_web_data_integrated'] = True
    return result


# モジュールレベルでのロガー設定（後方互換性）
import logging
logger = logging.getLogger('racing_system')

# デフォルト設定のインポート（後方互換性）
try:
    from racing_system.utils import DEFAULT_CONFIG, DETAILED_ODDS_WIN_RATES
except ImportError:
    # フォールバック設定
    DEFAULT_CONFIG = {
        'logging': {'level': 'INFO'},
        'database': {'path': './data/racing.db'},
        'pdf_extractor': {},
        'analyzer': {'internet_features': {'enabled': False}},
        'model_paths': {}
    }
    
    DETAILED_ODDS_WIN_RATES = {
        (1.0, 1.2): 85.0,
        (1.2, 1.5): 70.0,
        (1.5, 2.0): 55.0,
        (2.0, 3.0): 40.0,
        (3.0, 5.0): 25.0,
        (5.0, 10.0): 15.0,
        (10.0, 20.0): 8.0,
        (20.0, 50.0): 4.0,
        (50.0, 100.0): 2.0,
        (100.0, float('inf')): 1.0
    }