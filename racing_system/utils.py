#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
競馬予想システム - ユーティリティモジュール

共通のユーティリティ関数を提供します。
"""

import os
import re
import hashlib
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Union, Any

from .exceptions import ConfigError, SecurityError

try:
    import yaml
except ImportError:
    yaml = None


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    辞書を再帰的にマージする
    
    Args:
        dict1: ベース辞書
        dict2: マージする辞書
        
    Returns:
        マージされた辞書
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def load_config(config_path: Union[str, Path]) -> Dict:
    """
    YAML設定ファイルを読み込む
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        設定辞書
        
    Raises:
        ConfigError: 設定ファイルの読み込みエラー
    """
    if yaml is None:
        raise ConfigError("PyYAML is required for config loading")
    
    config_path = Path(config_path)
    
    # パストラバーサルチェック
    if not validate_file_path(config_path):
        raise SecurityError(f"Unsafe config path: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if not isinstance(config, dict):
            raise ConfigError("Config file must contain a dictionary")
        
        return config
        
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in config file: {e}")
    except Exception as e:
        raise ConfigError(f"Error loading config: {e}")


def setup_logging(logging_config: Dict) -> logging.Logger:
    """
    ロギングを設定する
    
    Args:
        logging_config: ロギング設定
        
    Returns:
        ロガーインスタンス
    """
    level = logging_config.get('level', 'INFO')
    log_format = logging_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
    log_file = logging_config.get('file')
    
    # ログレベルの設定
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # ロガーの設定
    logger = logging.getLogger('racing_system')
    logger.setLevel(numeric_level)
    
    # 既存のハンドラーをクリア
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # コンソールハンドラー
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # ファイルハンドラー（指定されている場合）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_roi_from_range_data(odds: float, range_data: List[Dict]) -> float:
    """
    オッズ範囲データからROIを取得
    
    Args:
        odds: オッズ
        range_data: 範囲データのリスト
        
    Returns:
        ROI値
    """
    for data in range_data:
        range_min, range_max = data['range']
        if range_min <= odds <= range_max:
            return data.get('roi_win', 0.0)
    
    # デフォルト値
    return 0.5


def get_statistical_win_rate(odds: float, odds_config: Dict) -> Dict:
    """
    統計的勝率を取得
    
    Args:
        odds: オッズ
        odds_config: オッズ設定
        
    Returns:
        統計情報の辞書
    """
    odds_win_rates = odds_config.get('odds_win_rates', [])
    
    for data in odds_win_rates:
        range_min, range_max = data['range']
        if range_min <= odds <= range_max:
            return {
                'win_rate': data.get('win_rate', 0.01),
                'place_rate': data.get('place_rate', 0.03),
                'show_rate': data.get('show_rate', 0.05),
                'roi_win': data.get('roi_win', 0.5),
                'roi_show': data.get('roi_show', 0.8)
            }
    
    # デフォルト値
    return {
        'win_rate': 0.001,
        'place_rate': 0.01,
        'show_rate': 0.02,
        'roi_win': 0.4,
        'roi_show': 0.6
    }


def calculate_odds_based_expected_value(odds: float, stats: Dict) -> float:
    """
    オッズベースの期待値を計算
    
    Args:
        odds: オッズ
        stats: 統計情報
        
    Returns:
        期待値
    """
    win_rate = stats.get('win_rate', 0.01)
    return win_rate * odds


def calculate_file_hash(file_path: Union[str, Path]) -> Optional[str]:
    """
    ファイルのSHA-256ハッシュを計算
    
    Args:
        file_path: ファイルパス
        
    Returns:
        ハッシュ値 or None
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return None
        
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
        
    except Exception:
        return None


def normalize_score(score: Union[int, float], min_val: float = 0.0, max_val: float = 100.0) -> float:
    """
    スコアを正規化
    
    Args:
        score: 元のスコア
        min_val: 最小値
        max_val: 最大値
        
    Returns:
        正規化されたスコア
    """
    if score is None:
        return min_val
    
    try:
        score = float(score)
        return max(min_val, min(max_val, score))
    except (ValueError, TypeError):
        return min_val


def parse_japanese_date(date_str: str) -> Optional[date]:
    """
    日本語の日付文字列をパース
    
    Args:
        date_str: 日付文字列
        
    Returns:
        dateオブジェクト or None
    """
    if not date_str or not isinstance(date_str, str):
        return None
    
    # 日本語形式のパターン
    patterns = [
        r'(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日',
        r'(\d{4})/(\d{1,2})/(\d{1,2})',
        r'(\d{4})-(\d{1,2})-(\d{1,2})',
        r'(\d{4})年(\d{1,2})月(\d{1,2})日'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, date_str)
        if match:
            try:
                year, month, day = map(int, match.groups())
                return date(year, month, day)
            except ValueError:
                continue
    
    return None


def format_finish_time(seconds: float) -> str:
    """
    秒数をタイム形式にフォーマット
    
    Args:
        seconds: 秒数
        
    Returns:
        フォーマットされたタイム
    """
    if not isinstance(seconds, (int, float)) or seconds <= 0:
        return "--:--.-"
    
    minutes = int(seconds // 60)
    secs = seconds % 60
    
    return f"{minutes}:{secs:04.1f}"


def parse_finish_time(time_str: str) -> Optional[float]:
    """
    タイム文字列を秒数に変換
    
    Args:
        time_str: タイム文字列
        
    Returns:
        秒数 or None
    """
    if not time_str or not isinstance(time_str, str):
        return None
    
    # パターン: "1:23.4" または "83.4"
    time_patterns = [
        r'(\d+):(\d+)\.(\d+)',  # 1:23.4
        r'(\d+)\.(\d+)',        # 83.4
        r'(\d+):(\d+)',         # 1:23
    ]
    
    for pattern in time_patterns:
        match = re.match(pattern, time_str.strip())
        if match:
            groups = match.groups()
            try:
                if len(groups) == 3:  # mm:ss.f
                    minutes, seconds, fraction = map(int, groups)
                    return minutes * 60 + seconds + fraction / 10
                elif len(groups) == 2:
                    if ':' in time_str:  # mm:ss
                        minutes, seconds = map(int, groups)
                        return minutes * 60 + seconds
                    else:  # ss.f
                        seconds, fraction = map(int, groups)
                        return seconds + fraction / 10
            except ValueError:
                continue
    
    return None


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    安全なfloat変換
    
    Args:
        value: 変換する値
        default: デフォルト値
        
    Returns:
        float値
    """
    if value is None:
        return default
    
    try:
        if isinstance(value, str):
            # カンマや空白を除去
            value = value.replace(',', '').strip()
            if not value:
                return default
        
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    安全なint変換
    
    Args:
        value: 変換する値
        default: デフォルト値
        
    Returns:
        int値
    """
    if value is None:
        return default
    
    try:
        if isinstance(value, str):
            # カンマや空白を除去
            value = value.replace(',', '').strip()
            if not value:
                return default
        
        return int(float(value))  # float経由で小数点以下を切り捨て
    except (ValueError, TypeError):
        return default


def validate_file_path(file_path: Union[str, Path]) -> bool:
    """
    ファイルパスの安全性を検証
    
    Args:
        file_path: ファイルパス
        
    Returns:
        安全かどうか
    """
    try:
        file_path = Path(file_path).resolve()
        
        # パストラバーサルチェック
        dangerous_patterns = ['..', '~', '$']
        path_str = str(file_path)
        
        for pattern in dangerous_patterns:
            if pattern in path_str:
                return False
        
        # 絶対パスであることを確認
        if not file_path.is_absolute():
            return False
        
        return True
        
    except Exception:
        return False