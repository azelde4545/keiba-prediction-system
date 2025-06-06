#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: racing_assistant.py
# Description: Claudeと連携する競馬予想アシスタント

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

# 競馬予想システムのインポート（パスを調整）
RACING_PATH = Path(r"C:\Users\setsu\Desktop\競馬用")
sys.path.append(str(RACING_PATH))

try:
    from keiba_predictor import KeibaPredictor
    from racing_system import display_enhanced_analysis_results, load_config
except ImportError as e:
    print(f"エラー: 競馬予想システムのインポートに失敗しました: {e}")
    print(f"パス {RACING_PATH} に競馬予想システムが存在するか確認してください。")
    sys.exit(1)

class RacingAssistant:
    """Claudeと連携する競馬予想アシスタント"""
    
    def __init__(self):
        """初期化"""
        # 競馬予想システムのパス
        self.racing_path = RACING_PATH
        
        # 設定ファイルパス
        self.config_path = self.racing_path / "config" / "predictor_config.yaml"
        if not self.config_path.exists():
            print(f"エラー: 設定ファイルが見つかりません: {self.config_path}")
            sys.exit(1)
        
        # 競馬予想システムの初期化
        try:
            self.predictor = KeibaPredictor(self.config_path)
            print("競馬予想システムの初期化が完了しました。")
        except Exception as e:
            print(f"エラー: 競馬予想システムの初期化に失敗しました: {e}")
            sys.exit(1)
    
    def predict_by_race_id(self, race_id: str, no_web: bool = False) -> Dict:
        """
        レースIDで予想を実行
        
        Args:
            race_id: レースID
            no_web: Web検索を無効にするか
            
        Returns:
            予想結果
        """
        if no_web:
            # Web検索を一時的に無効化
            original_web_scraper = self.predictor.web_scraper
            self.predictor.web_scraper = None
            self.predictor.analyzer.enable_internet_features = False
            print("Web検索機能を無効化しました。")
        
        try:
            result = self.predictor.analyze_race_id(race_id)
            formatted_result = self._format_result_for_claude(result)
            return formatted_result
        except Exception as e:
            return {"error": f"予想実行エラー: {e}"}
        finally:
            if no_web:
                # Web検索設定を元に戻す
                self.predictor.web_scraper = original_web_scraper
                self.predictor.analyzer.enable_internet_features = True
    
    def predict_by_pdf(self, pdf_path: Union[str, Path], force: bool = False, no_web: bool = False) -> Dict:
        """
        PDFファイルで予想を実行
        
        Args:
            pdf_path: PDFファイルのパス
            force: キャッシュを無視して強制的に再解析するか
            no_web: Web検索を無効にするか
            
        Returns:
            予想結果
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return {"error": f"PDFファイルが見つかりません: {pdf_path}"}
        
        if no_web:
            # Web検索を一時的に無効化
            original_web_scraper = self.predictor.web_scraper
            self.predictor.web_scraper = None
            self.predictor.analyzer.enable_internet_features = False
            print("Web検索機能を無効化しました。")
        
        try:
            result = self.predictor.analyze_pdf(pdf_path, force)
            formatted_result = self._format_result_for_claude(result)
            return formatted_result
        except Exception as e:
            return {"error": f"予想実行エラー: {e}"}
        finally:
            if no_web:
                # Web検索設定を元に戻す
                self.predictor.web_scraper = original_web_scraper
                self.predictor.analyzer.enable_internet_features = True
    
    def list_upcoming_races(self) -> List[Dict]:
        """
        今後のレース一覧を取得
        
        Returns:
            レース一覧
        """
        if not self.predictor.web_scraper:
            return [{"error": "Web検索機能が利用できないため、今後のレース情報を取得できません。"}]
        
        try:
            # 今日から1週間分のレースを取得
            today = datetime.now().date()
            races = []
            
            for i in range(7):
                target_date = today + timedelta(days=i)
                date_str = target_date.strftime('%Y-%m-%d')
                daily_races = self.predictor.web_scraper.get_race_schedule(date_str)
                
                if daily_races:
                    races.extend(daily_races)
            
            return races
        except Exception as e:
            return [{"error": f"レース情報取得エラー: {e}"}]
    
    def search_race(self, query: str) -> List[Dict]:
        """
        レース検索
        
        Args:
            query: 検索クエリ
            
        Returns:
            検索結果
        """
        if not self.predictor.web_scraper:
            return [{"error": "Web検索機能が利用できないため、レース検索を実行できません。"}]
        
        try:
            search_results = self.predictor.web_scraper.search_races(query)
            return search_results
        except Exception as e:
            return [{"error": f"レース検索エラー: {e}"}]
    
    def get_horse_info(self, horse_name: str) -> Dict:
        """
        馬情報の取得
        
        Args:
            horse_name: 馬名
            
        Returns:
            馬情報
        """
        if not self.predictor.web_scraper:
            return {"error": "Web検索機能が利用できないため、馬情報を取得できません。"}
        
        try:
            horse_info = self.predictor.web_scraper.get_horse_info(horse_name)
            return horse_info
        except Exception as e:
            return {"error": f"馬情報取得エラー: {e}"}
    
    def get_jockey_info(self, jockey_name: str) -> Dict:
        """
        騎手情報の取得
        
        Args:
            jockey_name: 騎手名
            
        Returns:
            騎手情報
        """
        if not self.predictor.web_scraper:
            return {"error": "Web検索機能が利用できないため、騎手情報を取得できません。"}
        
        try:
            jockey_info = self.predictor.web_scraper.get_jockey_info(jockey_name)
            return jockey_info
        except Exception as e:
            return {"error": f"騎手情報取得エラー: {e}"}
    
    def _format_result_for_claude(self, result: Dict) -> Dict:
        """
        予想結果をClaude用にフォーマット
        
        Args:
            result: 予想結果
            
        Returns:
            フォーマットされた結果
        """
        if 'error' in result:
            return result
        
        formatted = {
            "race_info": {
                "race_name": result.get("race_name", ""),
                "race_date": result.get("race_date", ""),
                "track": result.get("track", ""),
                "race_number": result.get("race_number", 0),
                "surface": result.get("surface", ""),
                "distance": result.get("distance", 0),
                "weather": result.get("weather", ""),
                "condition": result.get("condition", ""),
            },
            "predictions": []
        }
        
        # 予想結果を整形
        horses = result.get("analyzed_horses", [])
        for horse in horses:
            formatted["predictions"].append({
                "rank": horse.get("predicted_rank", 0),
                "horse_number": horse.get("horse_number", 0),
                "horse_name": horse.get("horse_name", ""),
                "jockey": horse.get("jockey", ""),
                "odds": horse.get("odds", 0.0),
                "win_probability": horse.get("win_probability", 0.0),
                "expected_value": horse.get("expected_value", 0.0),
                "score": horse.get("score", 0.0),
                "key_factors": horse.get("key_factors", []),
            })
        
        # 予想の種類と推奨馬券を追加
        if "recommendations" in result:
            formatted["recommendations"] = result["recommendations"]
        
        return formatted

def parse_arguments():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(description='競馬予想アシスタント')
    
    subparsers = parser.add_subparsers(dest='command', help='コマンド')
    
    # レースID予想コマンド
    race_parser = subparsers.add_parser('race', help='レースIDで予想')
    race_parser.add_argument('race_id', type=str, help='レースID')
    race_parser.add_argument('--no-web', action='store_true', help='Web検索を無効化')
    
    # PDF予想コマンド
    pdf_parser = subparsers.add_parser('pdf', help='PDFファイルで予想')
    pdf_parser.add_argument('pdf_path', type=str, help='PDFファイルのパス')
    pdf_parser.add_argument('--force', action='store_true', help='キャッシュを無視して強制的に再解析')
    pdf_parser.add_argument('--no-web', action='store_true', help='Web検索を無効化')
    
    # レース一覧コマンド
    list_parser = subparsers.add_parser('list', help='今後のレース一覧を表示')
    
    # レース検索コマンド
    search_parser = subparsers.add_parser('search', help='レース検索')
    search_parser.add_argument('query', type=str, help='検索クエリ')
    
    # 馬情報コマンド
    horse_parser = subparsers.add_parser('horse', help='馬情報の取得')
    horse_parser.add_argument('horse_name', type=str, help='馬名')
    
    # 騎手情報コマンド
    jockey_parser = subparsers.add_parser('jockey', help='騎手情報の取得')
    jockey_parser.add_argument('jockey_name', type=str, help='騎手名')
    
    return parser.parse_args()

def main():
    """メイン関数"""
    args = parse_arguments()
    
    # アシスタントの初期化
    assistant = RacingAssistant()
    
    if args.command == 'race':
        result = assistant.predict_by_race_id(args.race_id, args.no_web)
    elif args.command == 'pdf':
        result = assistant.predict_by_pdf(args.pdf_path, args.force, args.no_web)
    elif args.command == 'list':
        result = assistant.list_upcoming_races()
    elif args.command == 'search':
        result = assistant.search_race(args.query)
    elif args.command == 'horse':
        result = assistant.get_horse_info(args.horse_name)
    elif args.command == 'jockey':
        result = assistant.get_jockey_info(args.jockey_name)
    else:
        print("コマンドを指定してください。詳細は --help を参照してください。")
        return 1
    
    # 結果を表示（JSON形式）
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())