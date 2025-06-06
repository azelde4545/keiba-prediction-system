#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
競馬予想システム - 分析エンジンモジュール

レース分析と予測を行うコア機能を提供します。
"""

import json
import math
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Set, Callable
from collections import defaultdict, Counter

import numpy as np

from .exceptions import AnalysisError, ModelError, ValidationError
from .constants import Constants
from .utils import (
    normalize_score,
    get_statistical_win_rate,
    calculate_odds_based_expected_value,
    safe_int,
    safe_float,
    parse_japanese_date,
    format_finish_time
)
from .database import RacingDatabase

# 機械学習ライブラリのインポート試行
ML_LIBS_AVAILABLE = False
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    import joblib
    ML_LIBS_AVAILABLE = True
except ImportError:
    pass


class HorseRacingAnalyzer:
    """競馬レース分析エンジン"""
    
    def __init__(self, db: RacingDatabase, analyzer_cfg: Dict, 
                 model_paths: Dict, logger: logging.Logger, 
                 odds_stats_cfg: Optional[Dict] = None):
        """
        初期化
        
        Args:
            db: データベースインスタンス
            analyzer_cfg: 分析設定
            model_paths: モデルパス設定
            logger: ロガー
            odds_stats_cfg: オッズ統計設定
        """
        self.db = db
        self.analyzer_config = analyzer_cfg
        self.model_paths = model_paths
        self.logger = logger
        self.odds_stats_config = odds_stats_cfg or {}
        
        # 評価基準の重み付け設定
        self.weights = analyzer_cfg.get('weights', {
            'running_style_fit': 0.30,
            'track_surface': 0.18,
            'distance_fit': 0.18,
            'track_fit': 0.14,
            'recent_performance': 0.12,
            'track_condition_fit': 0.05,
            'jockey': 0.05,
            'pedigree': 0.02,
            'form': 0.08,
        })
        
        # 距離カテゴリ設定
        self.distance_categories = analyzer_cfg.get('distance_categories', {})
        
        # 血統キーワード読み込み
        self.pedigree_keywords = self._load_pedigree_keywords(
            analyzer_cfg.get('pedigree_keywords_path')
        )
        
        # その他の設定
        self.top_jockeys = analyzer_cfg.get('top_jockeys', [])
        self.recent_performance_races = analyzer_cfg.get('recent_performance_races', 5)
        self.recent_performance_weights = analyzer_cfg.get('recent_performance_weights', 
                                                         [0.4, 0.25, 0.15, 0.1, 0.1])
        self.jockey_stats_days = analyzer_cfg.get('jockey_stats_days', 365)
        self.form_ideal_interval = analyzer_cfg.get('form_ideal_interval', (28, 42))
        
        # デフォルト評価値
        self.evaluation_defaults = analyzer_cfg.get('evaluation_defaults', {})
        
        # インターネット機能設定
        internet_cfg = analyzer_cfg.get('internet_features', {})
        self.enable_internet_features = internet_cfg.get('enabled', False)
        self.use_simulated_internet_data = internet_cfg.get('use_simulated_data', False)
        
        # 機械学習モデルの読み込み
        self.ml_models = self._load_ml_models()
        
        # キャッシュ初期化
        self._running_style_cache = {}
        self._jockey_cache = {}
        self.race_track_bias_cache = None
        self._get_internet_data = None  # Web検索モジュール連携用フック
    
    def _load_pedigree_keywords(self, path: Optional[Union[str, Path]]) -> Dict:
        """血統キーワードを読み込む"""
        if not path:
            return {}
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load pedigree keywords from {path}: {e}")
            return {}
    
    def _load_ml_models(self) -> Dict:
        """機械学習モデルを読み込む"""
        models = {
            'classifier': None,
            'regressor': None,
            'scaler': None,
            'feature_columns': []
        }
        
        if not ML_LIBS_AVAILABLE:
            self.logger.warning("ML libraries not available. ML features disabled.")
            return models
        
        try:
            # 分類器
            classifier_path = Path(self.model_paths.get('classifier', ''))
            if classifier_path.exists():
                models['classifier'] = joblib.load(classifier_path)
                self.logger.info(f"Loaded classifier from {classifier_path}")
            
            # 回帰器
            regressor_path = Path(self.model_paths.get('regressor', ''))
            if regressor_path.exists():
                models['regressor'] = joblib.load(regressor_path)
                self.logger.info(f"Loaded regressor from {regressor_path}")
            
            # スケーラー
            scaler_path = Path(self.model_paths.get('scaler', ''))
            if scaler_path.exists():
                models['scaler'] = joblib.load(scaler_path)
                self.logger.info(f"Loaded scaler from {scaler_path}")
            
            # 特徴量カラム
            feature_cols_path = Path(self.model_paths.get('feature_columns', ''))
            if feature_cols_path.exists():
                with open(feature_cols_path, 'r') as f:
                    models['feature_columns'] = json.load(f)
                self.logger.info(f"Loaded feature columns from {feature_cols_path}")
        
        except Exception as e:
            self.logger.error(f"Error loading ML models: {e}", exc_info=True)
        
        return models
    
    def analyze_race(self, race_info: Dict) -> Dict:
        """レースを分析する"""
        try:
            # 入力検証
            if not isinstance(race_info, dict):
                raise ValidationError("Race info must be a dictionary")
            
            if 'horses' not in race_info or not race_info['horses']:
                raise ValidationError("No horses found in race info")
            
            # レース特性の分析
            race_characteristics = self._analyze_race_characteristics(race_info)
            
            # 各馬の分析
            analyzed_horses = []
            for horse in race_info['horses']:
                try:
                    analysis = self._analyze_horse(horse, race_info, race_characteristics)
                    analyzed_horses.append(analysis)
                except Exception as e:
                    self.logger.error(
                        f"Error analyzing horse {horse.get('horse_name', 'Unknown')}: {e}",
                        exc_info=True
                    )
            
            if not analyzed_horses:
                raise AnalysisError("No horses could be analyzed successfully")
            
            # スコアに基づいてソート
            analyzed_horses.sort(key=lambda x: x.get('final_score', 0), reverse=True)
            
            # 予想順位の割り当て
            for i, horse in enumerate(analyzed_horses, 1):
                horse['predicted_rank'] = i
            
            # 勝率の計算
            analyzed_horses = self._calculate_win_probabilities(analyzed_horses)
            
            # 期待値の計算
            recommended_horses = self._calculate_expected_values(analyzed_horses)
            
            # レース展開予想
            race_development = self._predict_race_development(
                analyzed_horses, race_characteristics
            )
            
            # 結果の構築
            result = {
                'race_info': race_info,
                'race_characteristics': race_characteristics,
                'analyzed_horses': analyzed_horses,
                'recommended_horses': recommended_horses,
                'race_development': race_development,
                'ml_available': self._is_ml_available(),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except (ValidationError, AnalysisError):
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in race analysis: {e}", exc_info=True)
            raise AnalysisError(f"Race analysis failed: {e}") from e
    
    def _analyze_race_characteristics(self, race_info: Dict) -> Dict:
        """レース特性を分析する"""
        surface = race_info.get('surface', '')
        distance = safe_int(race_info.get('distance'), 1600)
        track = race_info.get('track', '')
        track_condition = race_info.get('track_condition', '良')
        
        # 距離カテゴリの判定
        distance_category = self._get_distance_category(surface, distance)
        
        # 脚質分布の分析
        running_style_distribution = self._analyze_running_style_distribution(
            race_info.get('horses', [])
        )
        
        return {
            'surface': surface,
            'distance': distance,
            'distance_category': distance_category,
            'track': track,
            'track_condition': track_condition,
            'running_style_distribution': running_style_distribution,
            'expected_pace': self._estimate_pace(running_style_distribution),
            'horse_count': len(race_info.get('horses', []))
        }
    
    def _get_distance_category(self, surface: str, distance: int) -> str:
        """距離カテゴリを取得する"""
        categories = self.distance_categories.get(surface, {})
        
        if distance <= categories.get('short', 1399):
            return 'short'
        elif distance <= categories.get('mile', 1899):
            return 'mile'
        elif distance <= categories.get('middle', 2299):
            return 'middle'
        elif distance <= categories.get('mid_long', 2799):
            return 'mid_long'
        else:
            return 'long'
    
    def _analyze_running_style_distribution(self, horses: List[Dict]) -> Dict:
        """脚質分布を分析する"""
        distribution = defaultdict(int)
        
        for horse in horses:
            style = self._estimate_running_style(horse)
            distribution[style] += 1
        
        total = sum(distribution.values())
        if total == 0:
            return {}
        
        return {
            style: count / total 
            for style, count in distribution.items()
        }
    
    def _estimate_running_style(self, horse: Dict) -> str:
        """馬の脚質を推定する"""
        horse_id = horse.get('horse_id') or self.db._generate_horse_id(horse.get('horse_name'))
        
        if not horse_id:
            return Constants.RUNNING_STYLE_MID
        
        # キャッシュチェック
        if horse_id in self._running_style_cache:
            return self._running_style_cache[horse_id]
        
        # レース履歴から推定
        history = self.db.get_horse_race_history(horse_id, limit=5)
        
        if not history:
            # 血統から推定
            style = self._estimate_style_from_pedigree(horse)
            self._running_style_cache[horse_id] = style
            return style
        
        # デフォルト値を返す（実装簡略化）
        style = Constants.RUNNING_STYLE_MID
        self._running_style_cache[horse_id] = style
        return style
    
    def _estimate_style_from_pedigree(self, horse: Dict) -> str:
        """血統から脚質を推定する"""
        return Constants.RUNNING_STYLE_MID
    
    def _estimate_pace(self, style_distribution: Dict) -> str:
        """ペースを推定する"""
        front_ratio = style_distribution.get(Constants.RUNNING_STYLE_FRONT, 0)
        stalker_ratio = style_distribution.get(Constants.RUNNING_STYLE_STALKER, 0)
        
        if front_ratio >= 0.3 or (front_ratio + stalker_ratio) >= 0.5:
            return Constants.PACE_HIGH
        elif front_ratio <= 0.1:
            return Constants.PACE_SLOW
        else:
            return Constants.PACE_MEDIUM
    
    def _analyze_horse(self, horse: Dict, race_info: Dict, 
                      race_characteristics: Dict) -> Dict:
        """個別の馬を分析する"""
        # 馬IDの取得または生成
        horse_id = horse.get('horse_id')
        if not horse_id:
            horse_id = self.db._generate_horse_id(horse.get('horse_name'))
        
        # 基本情報のコピー
        analysis = horse.copy()
        analysis['horse_id'] = horse_id
        
        # 脚質の推定
        analysis['running_style'] = self._estimate_running_style(horse)
        
        # 各種評価
        scores = {}
        
        # 各種適性評価（簡略化）
        scores['running_style_fit'] = 65.0
        scores['surface_fit'] = 60.0
        scores['distance_fit'] = 60.0
        scores['track_fit'] = 60.0
        scores['track_condition_fit'] = 60.0
        scores['recent_performance'] = 50.0
        scores['jockey'] = 50.0
        scores['pedigree'] = 60.0
        scores['form'] = 60.0
        
        # インターネット機能による補正
        if self.enable_internet_features:
            self._apply_internet_adjustments(scores, horse)
        
        # 総合スコアの計算
        final_score = self._calculate_final_score(scores)
        
        # 結果の格納
        analysis['scores'] = scores
        analysis['final_score'] = final_score
        
        # 機械学習による予測
        if self._is_ml_available():
            ml_results = self._apply_ml_prediction(analysis, race_info)
            analysis.update(ml_results)
        
        return analysis
    
    def _apply_internet_adjustments(self, scores: Dict, horse: Dict):
        """インターネット機能による調整を適用する"""
        if self.use_simulated_internet_data:
            self._apply_simulated_adjustments(scores, horse)
        elif self._get_internet_data:
            self._apply_real_internet_adjustments(scores, horse)
    
    def _apply_simulated_adjustments(self, scores: Dict, horse: Dict):
        """シミュレートされた調整を適用する"""
        import random
        
        # 騎手の調子
        jockey_name = horse.get('jockey')
        if jockey_name:
            r = random.random()
            if r < 0.1:  # 10%の確率で好調
                scores['jockey'] = min(100.0, scores.get('jockey', 50.0) + 
                                     random.uniform(5.0, 15.0))
            elif r < 0.2:  # 10%の確率で不調
                scores['jockey'] = max(0.0, scores.get('jockey', 50.0) - 
                                     random.uniform(5.0, 15.0))
    
    def _apply_real_internet_adjustments(self, scores: Dict, horse: Dict):
        """実際のインターネットデータによる調整を適用する"""
        # 騎手の最近の調子
        jockey_name = horse.get('jockey')
        if jockey_name and self._get_internet_data:
            jockey_data = self._get_internet_data('jockey_recent_form', jockey_name)
            if jockey_data:
                if 'boost' in jockey_data:
                    scores['jockey'] = min(100.0, scores.get('jockey', 50.0) + 
                                         jockey_data['boost'])
                elif 'penalty' in jockey_data:
                    scores['jockey'] = max(0.0, scores.get('jockey', 50.0) - 
                                         jockey_data['penalty'])
    
    def _calculate_final_score(self, scores: Dict) -> float:
        """最終スコアを計算する"""
        total_score = 0.0
        total_weight = 0.0
        
        for key, weight in self.weights.items():
            if key in scores:
                score = normalize_score(scores[key])
                total_score += score * weight
                total_weight += weight
        
        if total_weight > 0:
            return total_score / total_weight
        else:
            return self.evaluation_defaults.get('final_score', 50.0)
    
    def _is_ml_available(self) -> bool:
        """機械学習が利用可能かチェックする"""
        return (
            ML_LIBS_AVAILABLE and
            self.ml_models['classifier'] is not None and
            self.ml_models['regressor'] is not None and
            self.ml_models['scaler'] is not None and
            len(self.ml_models['feature_columns']) > 0
        )
    
    def _apply_ml_prediction(self, horse_analysis: Dict, race_info: Dict) -> Dict:
        """機械学習による予測を適用する"""
        try:
            # 特徴量の準備
            features = self._prepare_ml_features(horse_analysis, race_info)
            
            if features is None:
                return {}
            
            # スケーリング
            features_scaled = self.ml_models['scaler'].transform([features])
            
            # 予測
            top3_prob = self.ml_models['classifier'].predict_proba(features_scaled)[0][1]
            predicted_time = self.ml_models['regressor'].predict(features_scaled)[0]
            
            return {
                'ml_top3_probability': top3_prob,
                'ml_predicted_time': predicted_time,
                'ml_features_used': True
            }
            
        except Exception as e:
            self.logger.error(f"ML prediction error: {e}", exc_info=True)
            return {}
    
    def _prepare_ml_features(self, horse_analysis: Dict, race_info: Dict) -> Optional[List[float]]:
        """機械学習用の特徴量を準備する"""
        feature_columns = self.ml_models['feature_columns']
        if not feature_columns:
            return None
        
        features = []
        
        for col in feature_columns:
            if col in horse_analysis:
                features.append(safe_float(horse_analysis[col], 0.0))
            elif col in horse_analysis.get('scores', {}):
                features.append(safe_float(horse_analysis['scores'][col], 0.0))
            else:
                features.append(0.0)
        
        return features
    
    def _calculate_win_probabilities(self, analyzed_horses: List[Dict]) -> List[Dict]:
        """勝率を計算する"""
        # 基本確率の計算
        total_score = sum(h.get('final_score', 0) for h in analyzed_horses)
        
        if total_score == 0:
            equal_prob = 1.0 / len(analyzed_horses)
            for horse in analyzed_horses:
                horse['base_probability'] = equal_prob
        else:
            for horse in analyzed_horses:
                horse['base_probability'] = horse.get('final_score', 0) / total_score
        
        # オッズベースの統計的勝率を考慮
        for horse in analyzed_horses:
            odds = horse.get('odds')
            if odds and odds > 0:
                stats = get_statistical_win_rate(odds, self.odds_stats_config)
                statistical_win_rate = stats.get('win_rate', 0.01)
            else:
                statistical_win_rate = 0.01
            
            # 基本確率と統計的勝率を組み合わせる
            base_prob = horse['base_probability']
            combined_prob = (base_prob * 0.7) + (statistical_win_rate * 0.3)
            
            # ML予測がある場合はさらに組み合わせる
            if 'ml_top3_probability' in horse:
                ml_win_prob = horse['ml_top3_probability'] * 0.33
                combined_prob = (combined_prob * 0.7) + (ml_win_prob * 0.3)
            
            horse['win_probability'] = combined_prob
        
        # 確率の正規化
        total_prob = sum(h['win_probability'] for h in analyzed_horses)
        if total_prob > 0:
            for horse in analyzed_horses:
                horse['win_probability'] /= total_prob
        
        return analyzed_horses
    
    def _calculate_expected_values(self, analyzed_horses: List[Dict]) -> List[Dict]:
        """期待値を計算する"""
        recommended = []
        
        for horse in analyzed_horses:
            odds = horse.get('odds')
            if not odds or odds <= 0:
                continue
            
            win_prob = horse.get('win_probability', 0)
            
            # 期待値計算
            expected_value = win_prob * odds
            
            # 統計的期待値も計算
            stats = get_statistical_win_rate(odds, self.odds_stats_config)
            statistical_ev = calculate_odds_based_expected_value(odds, stats)
            
            horse['expected_value'] = expected_value
            horse['statistical_ev'] = statistical_ev
            
            # 買い目推奨の判定
            if expected_value >= 1.05:
                horse['bet_recommended'] = True
                
                # 推奨強度の判定
                if expected_value >= 1.5:
                    horse['bet_strength'] = 'Strong'
                elif expected_value >= 1.2:
                    horse['bet_strength'] = 'Medium'
                else:
                    horse['bet_strength'] = 'Weak'
            else:
                horse['bet_recommended'] = False
                horse['bet_strength'] = None
            
            recommended.append(horse)
        
        # 期待値順にソート
        recommended.sort(key=lambda x: x.get('expected_value', 0), reverse=True)
        
        return recommended
    
    def _predict_race_development(self, analyzed_horses: List[Dict], 
                                race_characteristics: Dict) -> Dict:
        """レース展開を予想する"""
        try:
            # 脚質別に分類
            front_runners = []
            stalkers = []
            mid_runners = []
            closers = []
            
            for horse in analyzed_horses:
                style = horse.get('running_style')
                horse_num = horse.get('horse_number')
                
                if style == Constants.RUNNING_STYLE_FRONT:
                    front_runners.append(horse_num)
                elif style == Constants.RUNNING_STYLE_STALKER:
                    stalkers.append(horse_num)
                elif style == Constants.RUNNING_STYLE_MID:
                    mid_runners.append(horse_num)
                elif style == Constants.RUNNING_STYLE_CLOSER:
                    closers.append(horse_num)
            
            # ペース予想
            expected_pace = race_characteristics.get('expected_pace', Constants.PACE_MEDIUM)
            
            return {
                'front_runners': front_runners,
                'stalkers': stalkers,
                'mid_runners': mid_runners,
                'closers': closers,
                'expected_pace': expected_pace
            }
            
        except Exception as e:
            self.logger.error(f"Race development prediction error: {e}", exc_info=True)
            return {'error': str(e)}