#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
300円ベース投資戦略モジュール

効率的で責任ある300円ベース投資戦略を提供します。
倫理的配慮とリスク管理を重視した設計です。
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class InvestmentRecommendation:
    """投資推奨データクラス"""
    bet_type: str           # 券種
    combination: str        # 組み合わせ
    amount: int            # 投資額
    expected_return: float # 期待値
    risk_level: str        # リスクレベル
    confidence: float      # 信頼度


class Investment300Strategy:
    """300円ベース投資戦略クラス"""
    
    BASE_AMOUNT = 300
    MIN_BET_UNIT = 100  # 最小購入単位
    
    # リスクレベル別戦略定義
    STRATEGIES = {
        'conservative': {
            'name': '保守的戦略',
            'target_users': '初心者向け',
            'risk_level': '低',
            'allocations': {
                'win': 0.50,      # 単勝 50%
                'place': 0.30,    # 複勝 30%
                'quinella': 0.20  # 馬連 20%
            }
        },
        'balanced': {
            'name': 'バランス戦略',
            'target_users': '中級者向け',
            'risk_level': '中',
            'allocations': {
                'win': 0.30,         # 単勝 30%
                'exacta': 0.40,      # 馬単 40%
                'trifecta': 0.30     # 3連複 30%
            }
        },
        'aggressive': {
            'name': '積極戦略',
            'target_users': '上級者向け',
            'risk_level': '高',
            'allocations': {
                'exacta': 0.40,      # 馬単 40%
                'trifecta': 0.60     # 3連単 60%
            }
        }
    }
    
    def __init__(self):
        """初期化"""
        self.min_expected_value = 0.80  # 最小期待値閾値
        self.max_risk_horses = 3        # 最大投資対象馬数
    
    def calculate_investment_plan(self, 
                                horses: List[Dict], 
                                strategy: str = 'balanced') -> Dict:
        """
        300円ベース投資プランの計算
        
        Args:
            horses: 分析済み馬データのリスト
            strategy: 投資戦略 ('conservative', 'balanced', 'aggressive')
            
        Returns:
            投資プラン辞書
        """
        if strategy not in self.STRATEGIES:
            strategy = 'balanced'
        
        strategy_config = self.STRATEGIES[strategy]
        
        # 投資対象馬の選定（期待値ベース）
        viable_horses = self._select_viable_horses(horses)
        
        if not viable_horses:
            return {
                'strategy': strategy_config,
                'total_investment': 0,
                'recommendations': [],
                'warning': '投資推奨馬が見つかりません（期待値不足）'
            }
        
        # 券種別投資額計算
        allocations = strategy_config['allocations']
        recommendations = []
        
        for bet_type, ratio in allocations.items():
            amount = self._round_to_bet_unit(self.BASE_AMOUNT * ratio)
            if amount >= self.MIN_BET_UNIT:
                rec = self._create_recommendation(
                    bet_type, viable_horses, amount, strategy_config['risk_level']
                )
                if rec:
                    recommendations.append(rec)
        
        total_investment = sum(rec.amount for rec in recommendations)
        
        return {
            'strategy': strategy_config,
            'total_investment': total_investment,
            'recommendations': recommendations,
            'viable_horses': viable_horses,
            'risk_assessment': self._assess_overall_risk(recommendations),
            'educational_notes': self._generate_educational_notes(strategy, recommendations)
        }
    
    def _select_viable_horses(self, horses: List[Dict]) -> List[Dict]:
        """投資対象馬の選定"""
        viable = []
        
        for horse in horses:
            expected_value = horse.get('expected_value', 0.0)
            win_probability = horse.get('win_probability', 0.0)
            
            # 期待値と勝率による選定
            if (expected_value >= self.min_expected_value and 
                win_probability > 0.05):  # 最低5%の勝率
                viable.append(horse)
        
        # 期待値順でソートし、上位を選択
        viable.sort(key=lambda x: x.get('expected_value', 0), reverse=True)
        return viable[:self.max_risk_horses]
    
    def _create_recommendation(self, 
                             bet_type: str, 
                             horses: List[Dict], 
                             amount: int,
                             risk_level: str) -> Optional[InvestmentRecommendation]:
        """券種別推奨の作成"""
        if not horses:
            return None
        
        # 券種別組み合わせ生成
        if bet_type == 'win':
            # 単勝：最高期待値馬
            horse = horses[0]
            combination = f"{horse.get('horse_number')}番"
            expected_return = horse.get('expected_value', 0.0)
            confidence = min(horse.get('win_probability', 0.0) * 10, 1.0)
            
        elif bet_type == 'place':
            # 複勝：上位2頭
            top_horses = horses[:2]
            combination = '-'.join([str(h.get('horse_number')) for h in top_horses])
            expected_return = sum(h.get('expected_value', 0.0) for h in top_horses) / 2
            confidence = min(sum(h.get('win_probability', 0.0) for h in top_horses), 1.0)
            
        elif bet_type in ['quinella', 'exacta']:
            # 馬連・馬単：上位2頭の組み合わせ
            if len(horses) >= 2:
                combination = f"{horses[0].get('horse_number')}-{horses[1].get('horse_number')}"
                expected_return = (horses[0].get('expected_value', 0.0) + 
                                 horses[1].get('expected_value', 0.0)) / 2
                confidence = horses[0].get('win_probability', 0.0) * 0.7
            else:
                return None
                
        elif bet_type == 'trifecta':
            # 3連複・3連単：上位3頭
            if len(horses) >= 3:
                combination = f"{horses[0].get('horse_number')}-{horses[1].get('horse_number')}-{horses[2].get('horse_number')}"
                expected_return = sum(h.get('expected_value', 0.0) for h in horses[:3]) / 3
                confidence = horses[0].get('win_probability', 0.0) * 0.5
            else:
                return None
        else:
            return None
        
        return InvestmentRecommendation(
            bet_type=bet_type,
            combination=combination,
            amount=amount,
            expected_return=expected_return,
            risk_level=risk_level,
            confidence=confidence
        )
    
    def _round_to_bet_unit(self, amount: float) -> int:
        """投注単位に丸める"""
        return max(self.MIN_BET_UNIT, 
                  math.floor(amount / self.MIN_BET_UNIT) * self.MIN_BET_UNIT)
    
    def _assess_overall_risk(self, recommendations: List[InvestmentRecommendation]) -> Dict:
        """総合リスク評価"""
        if not recommendations:
            return {'level': 'なし', 'score': 0.0}
        
        avg_expected_return = sum(r.expected_return for r in recommendations) / len(recommendations)
        avg_confidence = sum(r.confidence for r in recommendations) / len(recommendations)
        
        # リスクスコア計算（低いほど安全）
        risk_score = max(0.0, 1.0 - (avg_expected_return * avg_confidence))
        
        if risk_score <= 0.3:
            level = '低'
        elif risk_score <= 0.6:
            level = '中'
        else:
            level = '高'
        
        return {
            'level': level,
            'score': risk_score,
            'avg_expected_return': avg_expected_return,
            'avg_confidence': avg_confidence
        }
    
    def _generate_educational_notes(self, 
                                  strategy: str, 
                                  recommendations: List[InvestmentRecommendation]) -> List[str]:
        """教育的解説の生成"""
        notes = [
            f"💡 選択戦略：{self.STRATEGIES[strategy]['name']} ({self.STRATEGIES[strategy]['target_users']})",
            f"💰 基本投資額：{self.BASE_AMOUNT}円（責任ある小額投資）",
            "⚖️ リスク管理：期待値0.8以上の馬のみ選定",
            "📊 分散投資：複数券種でリスク分散"
        ]
        
        if not recommendations:
            notes.append("⚠️ 今回は投資推奨なし（安全第一の判断）")
        else:
            total_investment = sum(r.amount for r in recommendations)
            notes.append(f"📈 総投資額：{total_investment}円（基本額の{total_investment/self.BASE_AMOUNT:.1%}）")
        
        notes.extend([
            "🔴 重要：これは教育的分析です",
            "🔴 投資は自己責任で行ってください",
            "🔴 余裕資金の範囲内で参加してください"
        ])
        
        return notes


def display_investment_plan_300(horses: List[Dict], strategy: str = 'balanced') -> None:
    """
    300円ベース投資プランの表示
    
    Args:
        horses: 分析済み馬データ
        strategy: 投資戦略
    """
    investment = Investment300Strategy()
    plan = investment.calculate_investment_plan(horses, strategy)
    
    print(f"\n=== 300円ベース投資戦略 ===")
    print(f"戦略: {plan['strategy']['name']} ({plan['strategy']['target_users']})")
    print(f"リスクレベル: {plan['strategy']['risk_level']}")
    print(f"総投資額: {plan['total_investment']}円")
    
    if plan['recommendations']:
        print(f"\n--- 投資推奨 ---")
        for i, rec in enumerate(plan['recommendations'], 1):
            print(f"{i}. {rec.bet_type}: {rec.combination} "
                  f"({rec.amount}円, 期待値:{rec.expected_return:.2f}, "
                  f"信頼度:{rec.confidence:.1%})")
        
        risk = plan['risk_assessment']
        print(f"\n--- リスク評価 ---")
        print(f"総合リスクレベル: {risk['level']}")
        print(f"平均期待値: {risk['avg_expected_return']:.2f}")
        print(f"平均信頼度: {risk['avg_confidence']:.1%}")
    
    print(f"\n--- 学習ポイント ---")
    for note in plan['educational_notes']:
        print(note)


# 統合用の便利関数
def integrate_300_investment_to_display():
    """既存のdisplay.pyに300円投資機能を統合するためのパッチ"""
    import os
    import sys
    
    # racing_systemモジュールのパスを追加
    racing_path = os.path.dirname(os.path.abspath(__file__))
    if racing_path not in sys.path:
        sys.path.append(racing_path)
    
    try:
        from racing_system.display import display_enhanced_analysis_results
        
        # オリジナル関数をバックアップ
        original_display = display_enhanced_analysis_results
        
        def enhanced_display_with_investment(analysis_result: Dict):
            """300円投資機能付きの拡張表示"""
            # 元の表示を実行
            original_display(analysis_result)
            
            # 300円投資プランを追加表示
            horses = analysis_result.get('analyzed_horses', [])
            if horses:
                display_investment_plan_300(horses, 'balanced')
        
        # モジュールレベルで関数を置き換え
        import racing_system.display
        racing_system.display.display_enhanced_analysis_results = enhanced_display_with_investment
        
        print("✅ 300円ベース投資機能が正常に統合されました")
        return True
        
    except ImportError as e:
        print(f"❌ 統合エラー: {e}")
        return False


if __name__ == "__main__":
    # テスト実行
    print("300円ベース投資戦略モジュール - テスト実行")
    
    # サンプルデータでテスト
    sample_horses = [
        {
            'horse_number': 6,
            'horse_name': 'ロードクロンヌ',
            'expected_value': 0.85,
            'win_probability': 0.12
        },
        {
            'horse_number': 8,
            'horse_name': 'ブライアンセンス',
            'expected_value': 0.82,
            'win_probability': 0.10
        }
    ]
    
    display_investment_plan_300(sample_horses, 'balanced')