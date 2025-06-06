#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
競馬予想システム - 型定義

システム全体で使用されるデータ型を定義します。
"""

from typing import Dict, List, Optional, Union, Any, Tuple, NamedTuple
from dataclasses import dataclass
from datetime import date, datetime


@dataclass
class RaceInfo:
    """レース情報データクラス"""
    race_id: Optional[str] = None
    race_name: Optional[str] = None
    race_date: Optional[str] = None
    track: Optional[str] = None
    race_number: Optional[int] = None
    surface: Optional[str] = None
    distance: Optional[int] = None
    weather: Optional[str] = None
    track_condition: Optional[str] = None
    grade: Optional[str] = None
    start_time: Optional[str] = None
    horses: List[Dict] = None
    
    def __post_init__(self):
        if self.horses is None:
            self.horses = []


@dataclass
class HorseInfo:
    """馬情報データクラス"""
    horse_id: Optional[str] = None
    horse_name: Optional[str] = None
    horse_number: Optional[int] = None
    frame_number: Optional[int] = None
    jockey: Optional[str] = None
    weight: Optional[float] = None
    odds: Optional[float] = None
    popularity: Optional[int] = None
    horse_weight: Optional[int] = None
    weight_diff: Optional[int] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    trainer: Optional[str] = None
    father: Optional[str] = None
    mother: Optional[str] = None
    mother_father: Optional[str] = None


@dataclass
class AnalysisResult:
    """分析結果データクラス"""
    horse_id: str
    horse_name: str
    horse_number: int
    running_style: str
    scores: Dict[str, float]
    final_score: float
    predicted_rank: Optional[int] = None
    win_probability: Optional[float] = None
    expected_value: Optional[float] = None
    bet_recommended: bool = False
    key_factors: List[str] = None
    
    def __post_init__(self):
        if self.key_factors is None:
            self.key_factors = []


@dataclass
class RaceCharacteristics:
    """レース特性データクラス"""
    surface: str
    distance: int
    distance_category: str
    track: str
    track_condition: str
    running_style_distribution: Dict[str, float]
    expected_pace: str
    horse_count: int


@dataclass
class PredictionResult:
    """予測結果データクラス"""
    race_info: RaceInfo
    race_characteristics: RaceCharacteristics
    analyzed_horses: List[AnalysisResult]
    recommended_horses: List[Dict]
    race_development: Dict
    ml_available: bool
    analysis_timestamp: str


# 型エイリアス
RacingData = Dict[str, Any]
HorseData = Dict[str, Any]
ScoreData = Dict[str, float]
OddsData = Dict[str, Union[float, int]]
ValidationResult = Dict[str, Any]
MLFeatures = List[float]
MLPrediction = Dict[str, float]


class RaceStatistics(NamedTuple):
    """レース統計情報"""
    total_races: int
    avg_horse_count: float
    avg_distance: float
    surface_distribution: Dict[str, int]
    track_distribution: Dict[str, int]


class JockeyStatistics(NamedTuple):
    """騎手統計情報"""
    total_races: int
    wins: int
    seconds: int
    thirds: int
    win_ratio: float
    place_ratio: float
    top3_ratio: float


class HorseStatistics(NamedTuple):
    """馬統計情報"""
    total_races: int
    wins: int
    win_ratio: float
    avg_finish_position: float
    best_distance_category: str
    preferred_surface: str


# 設定系の型
ConfigDict = Dict[str, Any]
LogLevel = str
FilePath = Union[str, Path]
OptionalFilePath = Optional[FilePath]

# 日付系の型
DateLike = Union[str, date, datetime]
OptionalDate = Optional[DateLike]