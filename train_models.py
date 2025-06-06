#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: train_models.py
# Description: 競馬予想用の機械学習モデルを訓練・保存するスクリプト

import os
import sys
import json
import logging
import argparse
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train_models')

# デフォルトのモデルパス
DEFAULT_MODEL_DIR = Path('./models')
DEFAULT_DATASET_DIR = Path('./data/training')

# 特徴量のデフォルト設定
DEFAULT_FEATURES = [
    'horse_number', 'frame_number', 'weight', 'horse_weight', 'weight_diff',
    'odds', 'popularity', 'age', 'speed_score', 'stamina_score', 'surface_fit',
    'distance_fit', 'track_condition_fit', 'recent_performance', 'jockey_score',
    'running_style_fit', 'final_score'
]

def prepare_dataset(data_path: Optional[Union[str, Path]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    訓練データを準備する
    
    Args:
        data_path: 訓練データのパス (CSVまたはPickleファイル)
        
    Returns:
        X_train, X_test, y_train, y_test: 訓練データと評価データ
    """
    # データセット作成または読み込み
    if data_path and Path(data_path).exists():
        logger.info(f"データセットを読み込み中: {data_path}")
        
        # ファイル形式に応じて読み込み
        data_path = Path(data_path)
        if data_path.suffix.lower() == '.csv':
            df = pd.read_csv(data_path)
        elif data_path.suffix.lower() in ['.pkl', '.pickle']:
            df = pd.read_pickle(data_path)
        else:
            raise ValueError("サポートしていないファイル形式です。CSV または Pickle 形式を使用してください。")
    else:
        # デモ用のデータセットを生成
        logger.info("デモ用のサンプルデータセットを生成します")
        np.random.seed(42)
        n_samples = 1000
        
        # 特徴量生成
        horse_numbers = np.random.randint(1, 18, size=n_samples)
        frame_numbers = np.random.randint(1, 9, size=n_samples)
        weights = np.random.normal(55, 2, size=n_samples)
        horse_weights = np.random.normal(480, 30, size=n_samples)
        weight_diffs = np.random.normal(0, 5, size=n_samples)
        odds = np.random.exponential(10, size=n_samples) + 1.5
        popularities = np.random.randint(1, 18, size=n_samples)
        ages = np.random.randint(3, 8, size=n_samples)
        
        # 各種スコア (0-100)
        speed_scores = np.random.normal(60, 15, size=n_samples).clip(0, 100)
        stamina_scores = np.random.normal(60, 15, size=n_samples).clip(0, 100)
        surface_fits = np.random.normal(60, 15, size=n_samples).clip(0, 100)
        distance_fits = np.random.normal(60, 15, size=n_samples).clip(0, 100)
        track_condition_fits = np.random.normal(60, 15, size=n_samples).clip(0, 100)
        recent_performances = np.random.normal(60, 15, size=n_samples).clip(0, 100)
        jockey_scores = np.random.normal(60, 15, size=n_samples).clip(0, 100)
        running_style_fits = np.random.normal(60, 15, size=n_samples).clip(0, 100)
        
        # 総合評価 (競走能力)
        final_scores = (
            0.2 * speed_scores + 
            0.2 * stamina_scores + 
            0.15 * surface_fits + 
            0.15 * distance_fits + 
            0.1 * track_condition_fits + 
            0.1 * recent_performances + 
            0.05 * jockey_scores + 
            0.05 * running_style_fits
        ) + np.random.normal(0, 5, n_samples)
        
        # ターゲット変数：着順 (1-3位と4位以下)
        # オッズとスコアに基づいて確率を計算
        win_probs = 1.0 / (odds * 0.8) * (final_scores / 60) * (1 + np.random.normal(0, 0.3, n_samples))
        win_probs = win_probs / win_probs.sum() * n_samples
        
        # 結果をシミュレート（ランダム要素あり）
        race_results = np.zeros(n_samples)
        races = np.random.randint(0, 100, size=n_samples)  # レース識別子
        
        for race_id in np.unique(races):
            race_mask = (races == race_id)
            race_horses = np.where(race_mask)[0]
            if len(race_horses) > 0:
                # 各馬の勝利確率に比例した結果を生成
                race_probs = win_probs[race_mask] / win_probs[race_mask].sum() if win_probs[race_mask].sum() > 0 else None
                if race_probs is not None:
                    sorted_indices = np.random.choice(race_horses, size=len(race_horses), replace=False, p=race_probs)
                    for rank, idx in enumerate(sorted_indices, 1):
                        race_results[idx] = rank
        
        # 着順をクリップ (最大18頭)
        race_results = np.clip(race_results, 1, 18)
        
        # 順位を1-3位とそれ以外に分類
        top3_results = np.where(race_results <= 3, 1, 0)
        
        # 着差（秒）を計算
        time_diffs = np.zeros(n_samples)
        for race_id in np.unique(races):
            race_mask = (races == race_id)
            if np.any(race_mask):
                # 1位の馬の基準タイム
                base_time = np.random.normal(120, 5)  # 2分前後
                
                # 各順位による着差を計算
                for i in np.where(race_mask)[0]:
                    rank = race_results[i]
                    time_diffs[i] = base_time + (rank - 1) * np.random.normal(0.5, 0.2)  # 1順位ごとに平均0.5秒差
        
        # データフレーム作成
        df = pd.DataFrame({
            'horse_number': horse_numbers,
            'frame_number': frame_numbers,
            'weight': weights,
            'horse_weight': horse_weights,
            'weight_diff': weight_diffs,
            'odds': odds,
            'popularity': popularities,
            'age': ages,
            'speed_score': speed_scores,
            'stamina_score': stamina_scores,
            'surface_fit': surface_fits,
            'distance_fit': distance_fits,
            'track_condition_fit': track_condition_fits,
            'recent_performance': recent_performances,
            'jockey_score': jockey_scores,
            'running_style_fit': running_style_fits,
            'final_score': final_scores,
            'race_id': races,
            'rank': race_results,
            'top3': top3_results,
            'time_diff': time_diffs
        })
    
    # 特徴量とターゲット変数を分離
    X = df[DEFAULT_FEATURES]
    y_classif = df['top3']  # 上位3位に入るかの分類
    y_regress = df['time_diff']  # タイム差の回帰
    
    return X, y_classif, y_regress

def train_models(X: pd.DataFrame, y_classif: pd.Series, y_regress: pd.Series, 
                output_dir: Path = DEFAULT_MODEL_DIR) -> Tuple[Dict, Dict, Dict, List[str]]:
    """
    機械学習モデルを訓練する
    
    Args:
        X: 特徴量データフレーム
        y_classif: 分類用ターゲット（上位3位入賞）
        y_regress: 回帰用ターゲット（タイム差）
        output_dir: モデル保存先ディレクトリ
        
    Returns:
        classifier, regressor, scaler, feature_columns: 訓練済みモデルとメタデータ
    """
    # トレーニングセットとテストセットに分割
    X_train, X_test, y_classif_train, y_classif_test, y_regress_train, y_regress_test = train_test_split(
        X, y_classif, y_regress, test_size=0.25, random_state=42
    )
    
    # 特徴量のスケーリング
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 分類モデル（上位3位に入るかどうか）
    logger.info("分類モデル（上位3位予測）の訓練中...")
    classifier = RandomForestClassifier(
        n_estimators=100, 
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    classifier.fit(X_train_scaled, y_classif_train)
    
    # 分類モデルの評価
    y_classif_pred = classifier.predict(X_test_scaled)
    classif_accuracy = accuracy_score(y_classif_test, y_classif_pred)
    logger.info(f"分類モデルの精度: {classif_accuracy:.4f}")
    
    # 回帰モデル（着差予測）
    logger.info("回帰モデル（タイム差予測）の訓練中...")
    regressor = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    regressor.fit(X_train_scaled, y_regress_train)
    
    # 回帰モデルの評価
    y_regress_pred = regressor.predict(X_test_scaled)
    regress_rmse = np.sqrt(mean_squared_error(y_regress_test, y_regress_pred))
    logger.info(f"回帰モデルのRMSE: {regress_rmse:.4f}")
    
    # 特徴量の重要度表示
    feature_importances_classif = classifier.feature_importances_
    feature_importances_regress = regressor.feature_importances_
    
    logger.info("分類モデルの特徴量重要度:")
    for feature, importance in sorted(zip(X.columns, feature_importances_classif), key=lambda x: x[1], reverse=True):
        logger.info(f"  {feature}: {importance:.4f}")
    
    logger.info("回帰モデルの特徴量重要度:")
    for feature, importance in sorted(zip(X.columns, feature_importances_regress), key=lambda x: x[1], reverse=True):
        logger.info(f"  {feature}: {importance:.4f}")
    
    # モデル情報の整理
    classifier_info = {
        'model': classifier,
        'accuracy': classif_accuracy,
        'feature_importances': dict(zip(X.columns, feature_importances_classif.tolist())),
        'training_date': datetime.now().isoformat()
    }
    
    regressor_info = {
        'model': regressor,
        'rmse': regress_rmse,
        'feature_importances': dict(zip(X.columns, feature_importances_regress.tolist())),
        'training_date': datetime.now().isoformat()
    }
    
    scaler_info = {
        'model': scaler,
        'training_date': datetime.now().isoformat()
    }
    
    feature_columns = X.columns.tolist()
    
    return classifier_info, regressor_info, scaler_info, feature_columns

def save_models(classifier_info: Dict, regressor_info: Dict, scaler_info: Dict, 
               feature_columns: List[str], output_dir: Path = DEFAULT_MODEL_DIR) -> None:
    """
    訓練済みモデルを保存する
    
    Args:
        classifier_info: 分類モデル情報
        regressor_info: 回帰モデル情報
        scaler_info: スケーラー情報
        feature_columns: 特徴量カラム
        output_dir: 保存先ディレクトリ
    """
    # 出力ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # モデルの保存
    classifier_path = output_dir / 'rf_classifier.joblib'
    regressor_path = output_dir / 'gb_regressor.joblib'
    scaler_path = output_dir / 'scaler.joblib'
    feature_columns_path = output_dir / 'feature_columns.json'
    
    # モデルの保存
    joblib.dump(classifier_info['model'], classifier_path)
    joblib.dump(regressor_info['model'], regressor_path)
    joblib.dump(scaler_info['model'], scaler_path)
    
    # 特徴量カラムをJSON形式で保存
    with open(feature_columns_path, 'w', encoding='utf-8') as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=2)
    
    # メタデータの保存
    metadata = {
        'classifier': {k: v for k, v in classifier_info.items() if k != 'model'},
        'regressor': {k: v for k, v in regressor_info.items() if k != 'model'},
        'scaler': {k: v for k, v in scaler_info.items() if k != 'model'},
        'feature_columns': feature_columns,
        'timestamp': datetime.now().isoformat()
    }
    
    metadata_path = output_dir / 'model_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"モデル保存完了:")
    logger.info(f"  分類器: {classifier_path}")
    logger.info(f"  回帰器: {regressor_path}")
    logger.info(f"  スケーラー: {scaler_path}")
    logger.info(f"  特徴量カラム: {feature_columns_path}")
    logger.info(f"  メタデータ: {metadata_path}")

def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description='競馬予想用の機械学習モデルを訓練するスクリプト')
    parser.add_argument('--data', type=str, help='訓練データのパス（CSVまたはPickle形式）')
    parser.add_argument('--output', type=str, default=str(DEFAULT_MODEL_DIR), help='モデル出力ディレクトリ')
    return parser.parse_args()

def main():
    """メイン関数"""
    args = parse_args()
    
    # データセット準備
    X, y_classif, y_regress = prepare_dataset(args.data)
    
    # 出力ディレクトリの設定
    output_dir = Path(args.output)
    
    # モデル訓練
    classifier_info, regressor_info, scaler_info, feature_columns = train_models(
        X, y_classif, y_regress, output_dir
    )
    
    # モデル保存
    save_models(classifier_info, regressor_info, scaler_info, feature_columns, output_dir)
    
    logger.info("モデル訓練完了!")
    return 0

if __name__ == "__main__":
    sys.exit(main())