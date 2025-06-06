#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
競馬予想システム - コアモジュール

このパッケージは競馬予想システムのコア機能を提供します。
"""

from .exceptions import (
    RacingSystemError,
    PDFExtractionError,
    AnalysisError,
    ValidationError,
    DatabaseError,
    ConfigError
)

from .constants import Constants

from .utils import (
    merge_dicts,
    load_config,
    setup_logging,
    get_roi_from_range_data,
    get_statistical_win_rate,
    calculate_odds_based_expected_value,
    calculate_file_hash,
    normalize_score,
    parse_japanese_date,
    format_finish_time,
    parse_finish_time,
    safe_float,
    safe_int
)

from .database import RacingDatabase
from .pdf_parser import AdvancedPDFExtractor
from .analyzer import HorseRacingAnalyzer
from .display import (
    display_detailed_pdf_extraction,
    display_enhanced_analysis_results
)

__all__ = [
    # 例外
    'RacingSystemError',
    'PDFExtractionError',
    'AnalysisError',
    'ValidationError',
    'DatabaseError',
    'ConfigError',
    
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
    
    # メインクラス
    'RacingDatabase',
    'AdvancedPDFExtractor',
    'HorseRacingAnalyzer',
    
    # 表示関数
    'display_detailed_pdf_extraction',
    'display_enhanced_analysis_results'
]

__version__ = '3.2.0'
__author__ = 'Racing System Development Team'