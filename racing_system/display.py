#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
競馬予想システム - 表示モジュール

分析結果の表示機能を提供します。
"""

from datetime import date
from typing import Dict, Optional, Any

from .constants import Constants
from .utils import safe_int, safe_float

try:
    from tabulate import tabulate
except ImportError:
    # tabulateが利用できない場合のフォールバック
    def tabulate(data, headers=None, tablefmt='simple', numalign=None):
        """シンプルなtabulateの代替実装"""
        lines = []
        if headers:
            lines.append("\t".join(str(h) for h in headers))
        for row in data:
            lines.append("\t".join(str(cell) for cell in row))
        return "\n".join(lines)


def display_detailed_pdf_extraction(race_info: Dict, formatting_cfg: Optional[Dict] = None):
    """
    PDFから抽出した詳細情報を表示
    
    Args:
        race_info: レース情報辞書
        formatting_cfg: フォーマット設定
    """
    output = []
    val_res = race_info.get('validation_result', {})
    
    output.append("=== PDF抽出データ詳細 ===")
    output.append(f"検証状態: {val_res.get('result', 'UNKNOWN')}")
    
    if errs := val_res.get('errors'):
        output.append(f"エラー: {'; '.join(errs)}")
    
    if warns := val_res.get('warnings'):
        output.append(f"警告: {'; '.join(warns)}")
    
    # レース基本情報
    race_headers = ['項目', '値']
    race_data = [
        ['レース名', race_info.get('race_name', '-')],
        ['日付', race_info.get('race_date', '-')],
        ['開催', race_info.get('track', '-')],
        ['R番号', race_info.get('race_number', '-')],
        ['馬場', race_info.get('surface', '-')],
        ['距離', race_info.get('distance', '-')],
        ['馬場状態', race_info.get('track_condition', '-')],
        ['天候', race_info.get('weather', '-')],
        ['グレード', race_info.get('grade', '-')],
        ['発走時刻', race_info.get('start_time', '-')]
    ]
    
    output.append("\n--- レース基本情報 ---")
    table_fmt = (formatting_cfg or {}).get('table_format', 'grid')
    
    try:
        output.append(tabulate(race_data, headers=race_headers, tablefmt=table_fmt))
    except Exception:
        # tabulateが失敗した場合のフォールバック
        output.extend([f"{item}: {val}" for item, val in race_data])
    
    # 出走馬情報
    horses = race_info.get('horses', [])
    output.append(f"\n--- 出走馬情報 ({len(horses)}頭) ---")
    
    if horses:
        horse_headers = ['馬番', '馬名', '枠番', '騎手', '斤量', 'オッズ', '人気', 
                        '性別', '年齢', '調教師', '馬体重', '増減']
        horse_table = []
        
        # 現在の年を取得
        current_year = date.today().year
        for race_date_str in [race_info.get('race_date'), race_info.get('_current_race_date')]:
            if race_date_str:
                try:
                    from datetime import datetime
                    date_str = race_date_str.replace('年', '/').replace('月', '/').replace('日', '')
                    current_year = datetime.strptime(date_str, '%Y/%m/%d').year
                    break
                except:
                    pass
        
        # 馬番でソート
        sorted_horses = sorted(horses, key=lambda h: safe_int(h.get('horse_number'), 99))
        
        for h in sorted_horses:
            # 年齢計算
            age = None
            if h.get('birth_year') and isinstance(h.get('birth_year'), int):
                age = current_year - h.get('birth_year')
            
            # オッズ
            odds = safe_float(h.get('odds'))
            odds_str = f"{odds:.1f}" if odds is not None else '-'
            
            # 人気
            pop = safe_int(h.get('popularity'))
            pop_str = str(pop) if pop is not None else '-'
            
            # 馬体重
            weight = h.get('horse_weight')
            weight_str = str(weight) if weight is not None else '-'
            
            # 増減
            diff = h.get('weight_diff')
            diff_str = f"{diff:+}" if diff is not None else '-'
            
            horse_table.append([
                h.get('horse_number', '-'),
                h.get('horse_name', '-'),
                h.get('frame_number', '-'),
                h.get('jockey', '-'),
                h.get('weight', '-'),
                odds_str,
                pop_str,
                h.get('gender', '-'),
                age if age is not None else '-',
                h.get('trainer', '-'),
                weight_str,
                diff_str
            ])
        
        try:
            output.append(tabulate(horse_table, headers=horse_headers, 
                                 tablefmt=table_fmt, numalign="right"))
        except Exception:
            # フォールバック
            output.append("\t".join(horse_headers))
            output.extend(["\t".join(map(str, row)) for row in horse_table])
    else:
        output.append("出走馬データがありません。")
    
    print("\n".join(output))


def display_enhanced_analysis_results(analysis_result: Dict):
    """
    分析結果を拡張表示（すべての馬を表示）
    
    Args:
        analysis_result: 分析結果辞書
    """
    race_info = analysis_result.get('race_info', {})
    race_chars = analysis_result.get('race_characteristics', {})
    analyzed_horses = analysis_result.get('analyzed_horses', [])
    recommended = analysis_result.get('recommended_horses', [])
    race_dev = analysis_result.get('race_development', {})
    ml_available = analysis_result.get('ml_available', False)
    
    print(f"\n=== 予想結果: {race_info.get('race_name')} ===")
    print(f"コース: {race_info.get('surface')}{race_info.get('distance')}m "
          f"({race_info.get('track_condition', '良')}) - "
          f"{race_info.get('track')} {race_info.get('race_number')}R")
    
    # 脚質のマッピング
    style_map = {
        Constants.RUNNING_STYLE_FRONT: '逃げ',
        Constants.RUNNING_STYLE_STALKER: '先行',
        Constants.RUNNING_STYLE_MID: '差し',
        Constants.RUNNING_STYLE_CLOSER: '追込'
    }
    
    # 1. すべての馬の勝率順リスト
    win_prob_headers = ['順位', '馬番', '馬名', '騎手', 'オッズ', '勝率', 
                       'モデル確率', '走法', '適性スコア']
    win_prob_table = []
    
    win_prob_horses = sorted(analyzed_horses, 
                           key=lambda x: x.get('win_probability', 0.0), 
                           reverse=True)
    
    for i, h in enumerate(win_prob_horses, 1):
        odds = safe_float(h.get('odds'))
        odds_str = f"{odds:.1f}" if odds is not None else '-'
        win_prob = h.get('win_probability', 0.0) * 100.0
        
        model_prob = None
        if ml_available:
            model_prob = h.get('model_probability', 0.0) * 100.0
        model_prob_str = f"{model_prob:.1f}%" if model_prob is not None else '-'
        
        style = style_map.get(h.get('running_style', '-'), '-')
        final_score = h.get('final_score', 0.0)
        
        win_prob_table.append([
            i,
            h.get('horse_number', '-'),
            h.get('horse_name', ''),
            h.get('jockey', ''),
            odds_str,
            f"{win_prob:.1f}%",
            model_prob_str,
            style,
            f"{final_score:.1f}"
        ])
    
    print("\n=== 全馬勝率順リスト ===")
    try:
        print(tabulate(win_prob_table, headers=win_prob_headers, 
                      tablefmt='grid', numalign="right"))
    except Exception:
        # フォールバック
        print("\t".join(win_prob_headers))
        print("\n".join(["\t".join(map(str, row)) for row in win_prob_table]))
    
    # 2. すべての馬の期待値順リスト
    ev_headers = ['順位', '馬番', '馬名', '騎手', 'オッズ', '勝率', 
                  '期待値', '推奨', '統計期待値', '走法']
    ev_table = []
    
    ev_horses = sorted(recommended, 
                      key=lambda x: x.get('expected_value', 0.0), 
                      reverse=True)
    
    for i, h in enumerate(ev_horses, 1):
        odds = safe_float(h.get('odds'))
        odds_str = f"{odds:.1f}" if odds is not None else '-'
        win_prob = h.get('win_probability', 0.0) * 100.0
        ev = h.get('expected_value', 0.0)
        
        stat_ev = h.get('statistical_ev')
        stat_ev_str = f"{stat_ev:.2f}" if stat_ev is not None else '-'
        
        style = style_map.get(h.get('running_style', '-'), '-')
        
        strength_map = {
            'Strong': '★★★',
            'Medium': '★★',
            'Weak': '★',
            None: ''
        }
        bet_str = strength_map.get(h.get('bet_strength'), '')
        
        ev_table.append([
            i,
            h.get('horse_number', '-'),
            h.get('horse_name', ''),
            h.get('jockey', ''),
            odds_str,
            f"{win_prob:.1f}%",
            f"{ev:.2f}",
            bet_str,
            stat_ev_str,
            style
        ])
    
    print("\n=== 全馬期待値順リスト ===")
    try:
        print(tabulate(ev_table, headers=ev_headers, 
                      tablefmt='grid', numalign="right"))
    except Exception:
        # フォールバック
        print("\t".join(ev_headers))
        print("\n".join(["\t".join(map(str, row)) for row in ev_table]))
    
    # 3. レース展開予想
    print("\n=== レース展開予想 ===")
    if dev_error := race_dev.get('error'):
        print(f"予想エラー: {dev_error}")
    else:
        pace = race_dev.get('expected_pace', Constants.PACE_MEDIUM)
        pace_text = {
            Constants.PACE_HIGH: '速いペース',
            Constants.PACE_MEDIUM: '平均的なペース',
            Constants.PACE_SLOW: '遅いペース'
        }.get(pace, '不明')
        print(f"予想ペース: {pace_text}")
        
        front = race_dev.get('front_runners', [])
        stalkers = race_dev.get('stalkers', [])
        if front:
            print(f"逃げ馬予想 (馬番): {', '.join(map(str, sorted(front)))}")
        if stalkers:
            print(f"先行馬予想 (馬番): {', '.join(map(str, sorted(stalkers)))}")
        
        if laps := race_dev.get('predicted_lap_times'):
            print(f"予想タイム: {laps.get('total_formatted')} "
                  f"(前半: {laps.get('first_half_sec')}秒, "
                  f"後半: {laps.get('second_half_sec')}秒)")
    
    # 4. 買い目推奨
    bet_horses = [h for h in ev_horses if h.get('bet_recommended')]
    if bet_horses:
        print("\n=== 買い目推奨（単勝） ===")
        strength_map = {
            'Strong': '★★★ 強力推奨',
            'Medium': '★★ 推奨',
            'Weak': '★ 検討'
        }
        
        for h in bet_horses:
            strength = strength_map.get(h.get('bet_strength'), '検討')
            print(f"{h.get('horse_number', '-')}. {h.get('horse_name')}: "
                  f"{strength} (期待値: {h.get('expected_value', 0):.2f})")
    else:
        print("\n※推奨買い目なし (期待値1.05以上の馬なし)")