#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
競馬予想システム - カスタム例外定義

システム全体で使用される例外クラスを定義します。
"""

class RacingSystemError(Exception):
    """競馬予想システムの基底例外クラス"""
    pass


class PDFExtractionError(RacingSystemError):
    """PDF抽出に関連するエラー"""
    pass


class AnalysisError(RacingSystemError):
    """レース分析に関連するエラー"""
    pass


class ValidationError(RacingSystemError):
    """データ検証に関連するエラー"""
    pass


class DatabaseError(RacingSystemError):
    """データベース操作に関連するエラー"""
    pass


class ConfigError(RacingSystemError):
    """設定ファイルに関連するエラー"""
    pass


class WebScrapingError(RacingSystemError):
    """Webスクレイピングに関連するエラー"""
    pass


class ModelError(RacingSystemError):
    """機械学習モデルに関連するエラー"""
    pass


class SecurityError(RacingSystemError):
    """セキュリティに関連するエラー"""
    pass