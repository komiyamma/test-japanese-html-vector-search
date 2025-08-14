#!/usr/bin/env python3
"""
統合テスト実行スクリプト
全体フローの動作確認とパフォーマンステストを実行
"""

import sys
import os
import unittest
import time
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

def run_integration_tests():
    """統合テストを実行"""
    print("🚀 統合テストとエンドツーエンドテストを開始します...")
    print("=" * 60)
    
    # テストディスカバリーを使用してテストを検索
    test_dir = Path(__file__).parent
    loader = unittest.TestLoader()
    
    try:
        # test_integration.pyからテストを読み込み
        suite = loader.loadTestsFromName('test_integration', module=None)
        
        # テストランナーを設定
        runner = unittest.TextTestRunner(
            verbosity=2,
            stream=sys.stdout,
            buffer=True
        )
        
        # テスト実行時間を測定
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        # 結果サマリーを表示
        print("\n" + "=" * 60)
        print("📊 テスト結果サマリー")
        print("=" * 60)
        print(f"実行時間: {end_time - start_time:.2f}秒")
        print(f"実行テスト数: {result.testsRun}")
        print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
        print(f"失敗: {len(result.failures)}")
        print(f"エラー: {len(result.errors)}")
        
        if result.failures:
            print("\n❌ 失敗したテスト:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        
        if result.errors:
            print("\n💥 エラーが発生したテスト:")
            for test, traceback in result.errors:
                print(f"  - {test}")
        
        if result.wasSuccessful():
            print("\n✅ すべての統合テストが成功しました！")
            return 0
        else:
            print("\n❌ 一部のテストが失敗しました")
            return 1
            
    except ImportError as e:
        print(f"❌ テストモジュールのインポートに失敗しました: {e}")
        print("必要な依存関係がインストールされているか確認してください。")
        return 1
    except Exception as e:
        print(f"❌ 予期しないエラーが発生しました: {e}")
        return 1

def check_dependencies():
    """必要な依存関係をチェック"""
    print("🔍 依存関係をチェックしています...")
    
    required_modules = [
        'numpy',
        'sqlite3',
        'beautifulsoup4'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            if module == 'beautifulsoup4':
                import bs4
            else:
                __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module}")
            missing_modules.append(module)
    
    # sentence_transformersは別途チェック
    try:
        import sentence_transformers
        print(f"  ✅ sentence_transformers")
    except ImportError as e:
        print(f"  ⚠️ sentence_transformers (統合テストはモックを使用)")
        missing_modules.append('sentence_transformers')
    except Exception as e:
        print(f"  ⚠️ sentence_transformers (エラー: {e})")
        missing_modules.append('sentence_transformers')
    
    if missing_modules and 'sentence_transformers' in missing_modules and len(missing_modules) == 1:
        print("⚠️ sentence_transformersが利用できませんが、モックを使用してテストを継続します。")
        return True
    elif missing_modules:
        print(f"\n❌ 以下のモジュールが不足しています: {', '.join(missing_modules)}")
        print("pip install -r requirements.txt を実行してください。")
        return False
    
    print("✅ すべての依存関係が満たされています。")
    return True

def check_test_data():
    """テストデータの存在をチェック"""
    print("📁 テストデータをチェックしています...")
    
    test_data_dir = Path(__file__).parent / 'test_data'
    required_files = [
        'sample_page_1.html',
        'sample_page_2.html',
        'sample_page_3.html',
        'empty_page.html',
        'large_page.html',
        'invalid_encoding.html'
    ]
    
    missing_files = []
    
    for filename in required_files:
        filepath = test_data_dir / filename
        if filepath.exists():
            print(f"  ✅ {filename}")
        else:
            print(f"  ❌ {filename}")
            missing_files.append(filename)
    
    if missing_files:
        print(f"\n❌ 以下のテストファイルが不足しています: {', '.join(missing_files)}")
        return False
    
    print("✅ すべてのテストデータが揃っています。")
    return True

def main():
    """メイン関数"""
    print("🧪 日本語ベクトル検索システム - 統合テスト")
    print("=" * 60)
    
    # 依存関係をチェック
    if not check_dependencies():
        return 1
    
    print()
    
    # テストデータをチェック
    if not check_test_data():
        return 1
    
    print()
    
    # 統合テストを実行
    return run_integration_tests()

if __name__ == '__main__':
    sys.exit(main())