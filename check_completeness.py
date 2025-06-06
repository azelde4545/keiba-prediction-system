#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
競馬予想システム - プロジェクト完成度チェック
"""

import sys
from pathlib import Path
import importlib.util

def check_project_completeness():
    """プロジェクトの完成度をチェック"""
    project_root = Path(__file__).parent
    
    print("=" * 70)
    print("競馬予想システム - プロジェクト完成度チェック")
    print("=" * 70)
    
    checks = {
        "必須ファイル": [],
        "モジュール構造": [],
        "テスト": [],
        "設定・ドキュメント": [],
        "CI/CD": []
    }
    
    # 必須ファイルのチェック
    required_files = [
        ("README.md", "プロジェクト説明"),
        ("requirements.txt", "依存関係リスト"),
        ("racing_system.py", "メインシステム"),
        ("keiba_predictor.py", "予測エンジン"),
        ("racing_assistant.py", "アシスタント"),
        ("pdf_extractor.py", "PDF解析"),
        ("racing_web_scraper.py", "Web検索"),
        ("train_models.py", "モデル学習")
    ]
    
    print("\n1. 必須ファイルチェック:")
    for file, desc in required_files:
        path = project_root / file
        if path.exists():
            checks["必須ファイル"].append(f"✅ {file} - {desc}")
            print(f"  ✅ {file} - {desc}")
        else:
            checks["必須ファイル"].append(f"❌ {file} - {desc}")
            print(f"  ❌ {file} - {desc}")
    
    # モジュール構造のチェック
    print("\n2. モジュール構造チェック:")
    modules = [
        "racing_system/__init__.py",
        "racing_system/exceptions.py",
        "racing_system/constants.py", 
        "racing_system/utils.py",
        "racing_system/database.py",
        "racing_system/display.py",
        "racing_system/pdf_parser.py",
        "racing_system/analyzer.py",
        "racing_system/types.py"
    ]
    
    for module in modules:
        path = project_root / module
        if path.exists():
            checks["モジュール構造"].append(f"✅ {module}")
            print(f"  ✅ {module}")
        else:
            checks["モジュール構造"].append(f"❌ {module}")
            print(f"  ❌ {module}")
    
    # テストチェック
    print("\n3. テストファイルチェック:")
    test_files = [
        "tests/__init__.py",
        "tests/test_utils.py",
        "tests/test_database.py",
        "tests/test_pdf_parser.py",
        "tests/test_analyzer.py",
        "tests/test_display.py",
        "tests/run_tests.py"
    ]
    
    for test_file in test_files:
        path = project_root / test_file
        if path.exists():
            checks["テスト"].append(f"✅ {test_file}")
            print(f"  ✅ {test_file}")
        else:
            checks["テスト"].append(f"❌ {test_file}")
            print(f"  ❌ {test_file}")
    
    # 設定・ドキュメント
    print("\n4. 設定・ドキュメントチェック:")
    config_files = [
        ("config/settings.yaml", "アプリケーション設定"),
        ("config/model_config.yaml", "モデル設定"),
        (".coveragerc", "カバレッジ設定"),
        ("pytest.ini", "pytest設定")
    ]
    
    for file, desc in config_files:
        path = project_root / file
        if path.exists():
            checks["設定・ドキュメント"].append(f"✅ {file} - {desc}")
            print(f"  ✅ {file} - {desc}")
        else:
            checks["設定・ドキュメント"].append(f"❌ {file} - {desc}")
            print(f"  ❌ {file} - {desc}")
    
    # CI/CDチェック
    print("\n5. CI/CD設定チェック:")
    ci_files = [
        ".github/workflows/test.yml",
        "run_coverage.py"
    ]
    
    for file in ci_files:
        path = project_root / file
        if path.exists():
            checks["CI/CD"].append(f"✅ {file}")
            print(f"  ✅ {file}")
        else:
            checks["CI/CD"].append(f"❌ {file}")
            print(f"  ❌ {file}")
    
    # インポートチェック
    print("\n6. モジュールインポートチェック:")
    try:
        spec = importlib.util.spec_from_file_location(
            "racing_system", 
            project_root / "racing_system" / "__init__.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("  ✅ racing_systemモジュールのインポート成功")
    except Exception as e:
        print(f"  ❌ racing_systemモジュールのインポート失敗: {e}")
    
    # 完成度の計算
    total_checks = sum(len(v) for v in checks.values())
    passed_checks = sum(1 for v in checks.values() for item in v if "✅" in item)
    completion_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
    
    # 結果サマリー
    print("\n" + "=" * 70)
    print("完成度サマリー")
    print("=" * 70)
    print(f"総チェック項目: {total_checks}")
    print(f"合格項目: {passed_checks}")
    print(f"不合格項目: {total_checks - passed_checks}")
    print(f"完成度: {completion_rate:.1f}%")
    
    # 推奨事項
    if completion_rate < 100:
        print("\n推奨アクション:")
        for category, items in checks.items():
            failed_items = [item for item in items if "❌" in item]
            if failed_items:
                print(f"\n{category}:")
                for item in failed_items:
                    print(f"  - {item.replace('❌', '').strip()}を追加/修正")
    else:
        print("\n🎉 プロジェクトは完全に構成されています！")
    
    return completion_rate >= 90  # 90%以上で合格

if __name__ == "__main__":
    is_complete = check_project_completeness()
    sys.exit(0 if is_complete else 1)