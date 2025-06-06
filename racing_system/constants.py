#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
競馬予想システム - 定数定義

システム全体で使用される定数を定義します。
"""

from pathlib import Path


class Constants:
    """システム定数クラス"""
    
    # トラックコード
    TRACK_CODES = {
        '札幌': '01',
        '函館': '02',
        '福島': '03',
        '新潟': '04',
        '東京': '05',
        '中山': '06',
        '中京': '07',
        '京都': '08',
        '阪神': '09',
        '小倉': '10'
    }
    DEFAULT_TRACK_CODE = '00'
    
    # 脚質定数
    RUNNING_STYLE_FRONT = 'front'      # 逃げ
    RUNNING_STYLE_STALKER = 'stalker'  # 先行
    RUNNING_STYLE_MID = 'mid'          # 差し
    RUNNING_STYLE_CLOSER = 'closer'    # 追込
    
    # ペース定数
    PACE_HIGH = 'high'      # ハイペース
    PACE_MEDIUM = 'medium'  # 平均ペース
    PACE_SLOW = 'slow'      # スローペース
    
    # ファイルパス関連
    DEFAULT_CONFIG_PATH = Path("./config/predictor_config.yaml")
    DEFAULT_DB_PATH = Path("./data/racing.db")
    DEFAULT_MODEL_DIR = Path("./models")
    DEFAULT_CACHE_DIR = Path("./cache")
    DEFAULT_LOG_DIR = Path("./logs")
    
    # 安全なベースディレクトリ（セキュリティ用）
    SAFE_BASE_DIR = Path.cwd()
    
    # 最大値・最小値
    MIN_HORSE_NUMBER = 1
    MAX_HORSE_NUMBER = 18
    MIN_FRAME_NUMBER = 1
    MAX_FRAME_NUMBER = 8
    MIN_WEIGHT = 40.0
    MAX_WEIGHT = 70.0
    MIN_HORSE_WEIGHT = 300
    MAX_HORSE_WEIGHT = 700
    MIN_AGE = 2
    MAX_AGE = 15
    MIN_DISTANCE = 800
    MAX_DISTANCE = 5000
    
    # スコアのデフォルト値
    DEFAULT_SCORE = 60.0
    MIN_SCORE = 0.0
    MAX_SCORE = 100.0
    
    # その他の定数
    MAX_RECENT_RACES = 10
    CACHE_EXPIRY_HOURS = 24
    REQUEST_TIMEOUT = 30
    MAX_RETRY_COUNT = 3