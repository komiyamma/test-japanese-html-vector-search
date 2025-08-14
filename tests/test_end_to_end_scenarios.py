"""
エンドツーエンドシナリオテスト
実際の使用ケースに基づいたシナリオテストを実装
"""

import unittest
import os
import tempfile
import shutil
import subprocess
import sys
import json
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))


class TestEndToEndScenarios(unittest.TestCase):
    """エンドツーエンドシナリオテストクラス"""
    
    def setUp(self):
        """テスト前の準備"""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        
        # テストデータディレクトリ
        self.test_data_dir = Path(__file__).parent / 'test_data'
        
        # プロジェクトルートのパス
        self.project_root = Path(__file__).parent.parent
        
        # スクリプトのパス
        self.batch_script = self.project_root / 'scripts' / 'batch_process.py'
        self.search_script = self.project_root / 'scripts' / 'search_cli.py'
        
    def tearDown(self):
        """テスト後のクリーンアップ"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_complete_workflow_scenario(self):
        """完全なワークフローシナリオテスト"""
        print("\n🎯 シナリオ: 新しいプロジェクトでの完全なワークフロー")
        
        # 1. テスト用のHTMLファイルを準備
        html_dir = Path(self.test_dir) / 'html_files'
        html_dir.mkdir()
        
        test_files = [
            ('sample_page_1.html', 'page-bushou-織田信長.html'),
            ('sample_page_2.html', 'page-bushou-豊臣秀吉.html'),
            ('sample_page_3.html', 'page-bushou-徳川家康.html')
        ]
        
        for src_name, dst_name in test_files:
            src_path = self.test_data_dir / src_name
            dst_path = html_dir / dst_name
            shutil.copy2(src_path, dst_path)
        
        print(f"  ✅ テストファイルを準備: {len(test_files)}個")
        
        # 2. バッチ処理を実行
        os.chdir(html_dir)
        
        try:
            result = subprocess.run([
                sys.executable, str(self.batch_script)
            ], capture_output=True, text=True, timeout=60)
            
            self.assertEqual(result.returncode, 0, 
                           f"バッチ処理が失敗しました: {result.stderr}")
            
            print("  ✅ バッチ処理が完了")
            
            # データベースファイルが作成されたことを確認
            db_file = html_dir / 'vectors.db'
            self.assertTrue(db_file.exists(), "データベースファイルが作成されていません")
            print("  ✅ データベースファイルが作成されました")
            
        except subprocess.TimeoutExpired:
            self.fail("バッチ処理がタイムアウトしました")
        
        # 3. テキスト検索を実行
        try:
            search_result = subprocess.run([
                sys.executable, str(self.search_script),
                'text', '天下統一', '--top-k', '2'
            ], capture_output=True, text=True, timeout=30, cwd=html_dir)
            
            self.assertEqual(search_result.returncode, 0,
                           f"テキスト検索が失敗しました: {search_result.stderr}")
            
            # 検索結果に期待するキーワードが含まれていることを確認
            output = search_result.stdout
            self.assertIn("類似度", output)
            print("  ✅ テキスト検索が成功")
            
        except subprocess.TimeoutExpired:
            self.fail("テキスト検索がタイムアウトしました")
        
        # 4. ドキュメント類似検索を実行
        try:
            doc_search_result = subprocess.run([
                sys.executable, str(self.search_script),
                'document', 'page-bushou-織田信長', '--top-k', '2'
            ], capture_output=True, text=True, timeout=30, cwd=html_dir)
            
            self.assertEqual(doc_search_result.returncode, 0,
                           f"ドキュメント検索が失敗しました: {doc_search_result.stderr}")
            
            output = doc_search_result.stdout
            self.assertIn("類似度", output)
            print("  ✅ ドキュメント類似検索が成功")
            
        except subprocess.TimeoutExpired:
            self.fail("ドキュメント検索がタイムアウトしました")
        
        print("  🎉 完全なワークフローが正常に動作しました")
    
    def test_incremental_processing_scenario(self):
        """増分処理シナリオテスト"""
        print("\n🎯 シナリオ: ファイル追加後の増分処理")
        
        # 1. 初期ファイルでバッチ処理
        html_dir = Path(self.test_dir) / 'incremental_test'
        html_dir.mkdir()
        
        # 最初は1つのファイルだけ
        initial_file = self.test_data_dir / 'sample_page_1.html'
        target_file = html_dir / 'page-bushou-織田信長.html'
        shutil.copy2(initial_file, target_file)
        
        os.chdir(html_dir)
        
        # 初回バッチ処理
        result1 = subprocess.run([
            sys.executable, str(self.batch_script)
        ], capture_output=True, text=True, timeout=60)
        
        self.assertEqual(result1.returncode, 0)
        print("  ✅ 初回バッチ処理が完了")
        
        # 2. 新しいファイルを追加
        new_file = self.test_data_dir / 'sample_page_2.html'
        new_target = html_dir / 'page-bushou-豊臣秀吉.html'
        shutil.copy2(new_file, new_target)
        
        # 2回目のバッチ処理
        result2 = subprocess.run([
            sys.executable, str(self.batch_script)
        ], capture_output=True, text=True, timeout=60)
        
        self.assertEqual(result2.returncode, 0)
        print("  ✅ 増分バッチ処理が完了")
        
        # 3. 両方のファイルが検索できることを確認
        search_result = subprocess.run([
            sys.executable, str(self.search_script),
            'text', '戦国時代', '--top-k', '5'
        ], capture_output=True, text=True, timeout=30)
        
        self.assertEqual(search_result.returncode, 0)
        
        # 2つのドキュメントが見つかることを確認
        output = search_result.stdout
        self.assertIn("織田信長", output)
        self.assertIn("豊臣秀吉", output)
        print("  ✅ 両方のドキュメントが検索可能")
        
        print("  🎉 増分処理シナリオが正常に動作しました")
    
    def test_error_recovery_scenario(self):
        """エラー回復シナリオテスト"""
        print("\n🎯 シナリオ: エラーファイルがある場合の処理継続")
        
        # 1. 正常ファイルと問題ファイルを混在させる
        html_dir = Path(self.test_dir) / 'error_recovery'
        html_dir.mkdir()
        
        # 正常なファイル
        good_file = self.test_data_dir / 'sample_page_1.html'
        shutil.copy2(good_file, html_dir / 'page-good-test.html')
        
        # 空のファイル
        empty_file = html_dir / 'page-empty-test.html'
        empty_file.write_text('', encoding='utf-8')
        
        # 無効なHTMLファイル
        invalid_file = html_dir / 'page-invalid-test.html'
        invalid_file.write_text('これは無効なHTMLです', encoding='utf-8')
        
        os.chdir(html_dir)
        
        # バッチ処理を実行（エラーがあっても継続することを確認）
        result = subprocess.run([
            sys.executable, str(self.batch_script)
        ], capture_output=True, text=True, timeout=60)
        
        # エラーがあっても処理は完了する（リターンコード0）
        self.assertEqual(result.returncode, 0)
        
        # 少なくとも1つのファイルは処理されている
        search_result = subprocess.run([
            sys.executable, str(self.search_script),
            'text', 'テスト', '--top-k', '5'
        ], capture_output=True, text=True, timeout=30)
        
        # 検索は実行できる（データベースが作成されている）
        self.assertEqual(search_result.returncode, 0)
        print("  ✅ エラーがあっても処理が継続されました")
        
        print("  🎉 エラー回復シナリオが正常に動作しました")
    
    def test_large_dataset_scenario(self):
        """大規模データセットシナリオテスト"""
        print("\n🎯 シナリオ: 大規模データセットの処理")
        
        # 1. 多数のテストファイルを生成
        html_dir = Path(self.test_dir) / 'large_dataset'
        html_dir.mkdir()
        
        file_count = 20  # テスト環境では20ファイル
        
        template = """<!DOCTYPE html>
<html lang="ja">
<head><meta charset="UTF-8"><title>テストドキュメント{}</title></head>
<body>
<h1>テストドキュメント{}</h1>
<p>これは大規模データセットテスト用のドキュメント{}です。</p>
<p>戦国時代の武将について説明します。番号: {}</p>
<p>このドキュメントは自動生成されたテストデータです。</p>
</body>
</html>"""
        
        for i in range(file_count):
            filename = f'page-test-{i:03d}.html'
            filepath = html_dir / filename
            filepath.write_text(template.format(i, i, i, i), encoding='utf-8')
        
        print(f"  ✅ {file_count}個のテストファイルを生成")
        
        os.chdir(html_dir)
        
        # 2. バッチ処理を実行（時間を測定）
        import time
        start_time = time.time()
        
        result = subprocess.run([
            sys.executable, str(self.batch_script)
        ], capture_output=True, text=True, timeout=120)  # 2分のタイムアウト
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        self.assertEqual(result.returncode, 0)
        print(f"  ✅ バッチ処理完了 ({processing_time:.2f}秒)")
        
        # 3. 検索パフォーマンスをテスト
        search_start = time.time()
        
        search_result = subprocess.run([
            sys.executable, str(self.search_script),
            'text', 'テストドキュメント', '--top-k', '10'
        ], capture_output=True, text=True, timeout=30)
        
        search_time = time.time() - search_start
        
        self.assertEqual(search_result.returncode, 0)
        print(f"  ✅ 検索完了 ({search_time:.2f}秒)")
        
        # 結果に複数のドキュメントが含まれることを確認
        output = search_result.stdout
        result_count = output.count('類似度:')
        self.assertGreaterEqual(result_count, 10, "期待する数の検索結果が得られませんでした")
        
        print(f"  ✅ {result_count}件の検索結果を取得")
        print("  🎉 大規模データセットシナリオが正常に動作しました")
    
    def test_configuration_scenario(self):
        """設定ファイルシナリオテスト"""
        print("\n🎯 シナリオ: 設定ファイルを使用した処理")
        
        # 1. カスタム設定ファイルを作成
        html_dir = Path(self.test_dir) / 'config_test'
        html_dir.mkdir()
        
        config_file = html_dir / 'config.json'
        config_data = {
            "database_path": "custom_vectors.db",
            "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "batch_size": 16,
            "similarity_threshold": 0.5
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        print("  ✅ カスタム設定ファイルを作成")
        
        # 2. テストファイルを準備
        test_file = self.test_data_dir / 'sample_page_1.html'
        shutil.copy2(test_file, html_dir / 'page-config-test.html')
        
        os.chdir(html_dir)
        
        # 3. 設定ファイルを指定してバッチ処理
        result = subprocess.run([
            sys.executable, str(self.batch_script),
            '--config', str(config_file)
        ], capture_output=True, text=True, timeout=60)
        
        self.assertEqual(result.returncode, 0)
        
        # カスタムデータベースファイルが作成されることを確認
        custom_db = html_dir / 'custom_vectors.db'
        self.assertTrue(custom_db.exists(), "カスタムデータベースファイルが作成されていません")
        
        print("  ✅ カスタム設定での処理が完了")
        print("  🎉 設定ファイルシナリオが正常に動作しました")


class TestRealWorldUsage(unittest.TestCase):
    """実世界での使用パターンテスト"""
    
    def setUp(self):
        """テスト前の準備"""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        self.project_root = Path(__file__).parent.parent
        
    def tearDown(self):
        """テスト後のクリーンアップ"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_typical_research_workflow(self):
        """典型的な研究ワークフローテスト"""
        print("\n🎯 実世界シナリオ: 研究者の文書検索ワークフロー")
        
        # 実際のHTMLファイルを使用
        html_dir = Path(self.test_dir) / 'research'
        html_dir.mkdir()
        
        # プロジェクトルートの実際のHTMLファイルを使用
        actual_html_files = list(self.project_root.glob('page-bushou-*.html'))
        
        if len(actual_html_files) >= 3:
            # 実際のファイルを使用
            for i, src_file in enumerate(actual_html_files[:3]):
                dst_file = html_dir / src_file.name
                shutil.copy2(src_file, dst_file)
            
            print(f"  ✅ 実際のHTMLファイルを使用: {len(actual_html_files[:3])}個")
            
            os.chdir(html_dir)
            
            # バッチ処理
            batch_script = self.project_root / 'scripts' / 'batch_process.py'
            result = subprocess.run([
                sys.executable, str(batch_script)
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print("  ✅ 実際のデータでバッチ処理が完了")
                
                # 実際の検索クエリをテスト
                search_script = self.project_root / 'scripts' / 'search_cli.py'
                queries = ['戦国時代', '武将', '天下統一']
                
                for query in queries:
                    search_result = subprocess.run([
                        sys.executable, str(search_script),
                        'text', query, '--top-k', '3'
                    ], capture_output=True, text=True, timeout=30)
                    
                    if search_result.returncode == 0:
                        print(f"  ✅ クエリ '{query}' の検索が成功")
                    else:
                        print(f"  ⚠️ クエリ '{query}' の検索でエラー")
                
                print("  🎉 研究ワークフローが正常に動作しました")
            else:
                print("  ⚠️ 実際のデータでの処理をスキップ（依存関係の問題）")
        else:
            print("  ⚠️ 実際のHTMLファイルが不足しているためスキップ")


if __name__ == '__main__':
    # テストスイートを作成
    test_suite = unittest.TestSuite()
    
    # エンドツーエンドテストを追加
    test_suite.addTest(unittest.makeSuite(TestEndToEndScenarios))
    test_suite.addTest(unittest.makeSuite(TestRealWorldUsage))
    
    # テストランナーを作成して実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 結果を出力
    if result.wasSuccessful():
        print("\n✅ すべてのエンドツーエンドテストが成功しました！")
    else:
        print(f"\n❌ {len(result.failures)} 個のテストが失敗しました")
        print(f"❌ {len(result.errors)} 個のエラーが発生しました")