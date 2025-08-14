"""
BatchProcessorクラスの単体テスト

このモジュールは、BatchProcessorクラスの各機能をテストします。
"""

import unittest
import tempfile
import os
import shutil
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# テスト対象のインポート
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.batch_processor import BatchProcessor, ProcessingResult
    from src.html_processor import HTMLProcessor
    from src.vector_embedder import VectorEmbedder
    from src.database_manager import DatabaseManager
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("依存関係が不足している可能性があります。")
    sys.exit(1)


class TestBatchProcessor(unittest.TestCase):
    """BatchProcessorクラスのテストケース"""
    
    def setUp(self):
        """各テストの前に実行される初期化処理"""
        # 一時ディレクトリを作成
        self.temp_dir = tempfile.mkdtemp()
        self.temp_db_path = os.path.join(self.temp_dir, "test_vectors.db")
        
        # テスト用HTMLファイルを作成
        self.create_test_html_files()
        
        # モックオブジェクトを作成
        self.mock_html_processor = Mock(spec=HTMLProcessor)
        self.mock_vector_embedder = Mock(spec=VectorEmbedder)
        self.mock_database_manager = Mock(spec=DatabaseManager)
        
        # BatchProcessorインスタンスを作成
        self.batch_processor = BatchProcessor(
            html_processor=self.mock_html_processor,
            vector_embedder=self.mock_vector_embedder,
            database_manager=self.mock_database_manager,
            base_directory=self.temp_dir
        )
    
    def tearDown(self):
        """各テストの後に実行されるクリーンアップ処理"""
        # 一時ディレクトリを削除
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_html_files(self):
        """テスト用のHTMLファイルを作成"""
        test_files = [
            "page-bushou-徳川家康.html",
            "page-bushou-織田信長.html",
            "page-bushou-豊臣秀吉.html",
            "other-file.html",  # パターンに一致しないファイル
            "page-test.txt"     # HTMLファイルではない
        ]
        
        for filename in test_files:
            file_path = os.path.join(self.temp_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"""
                <html>
                <head><title>{filename}</title></head>
                <body>
                    <h1>テストページ: {filename}</h1>
                    <p>これは{filename}のテスト用コンテンツです。</p>
                </body>
                </html>
                """)
    
    def test_init(self):
        """初期化のテスト"""
        # デフォルト初期化
        processor = BatchProcessor(base_directory=self.temp_dir)
        self.assertIsInstance(processor.html_processor, HTMLProcessor)
        self.assertIsInstance(processor.vector_embedder, VectorEmbedder)
        self.assertIsInstance(processor.database_manager, DatabaseManager)
        self.assertEqual(str(processor.base_directory), self.temp_dir)
        
        # カスタム初期化
        self.assertEqual(self.batch_processor.html_processor, self.mock_html_processor)
        self.assertEqual(self.batch_processor.vector_embedder, self.mock_vector_embedder)
        self.assertEqual(self.batch_processor.database_manager, self.mock_database_manager)
    
    def test_discover_html_files(self):
        """HTMLファイル発見機能のテスト"""
        # デフォルトパターンでの発見
        files = self.batch_processor.discover_html_files()
        
        # page-*.htmlパターンに一致するファイルのみが発見されることを確認
        expected_files = [
            "page-bushou-徳川家康.html",
            "page-bushou-織田信長.html", 
            "page-bushou-豊臣秀吉.html"
        ]
        
        found_basenames = [os.path.basename(f) for f in files]
        found_basenames.sort()
        expected_files.sort()
        
        self.assertEqual(found_basenames, expected_files)
        self.assertEqual(len(files), 3)
        
        # カスタムパターンでの発見
        all_html_files = self.batch_processor.discover_html_files("*.html")
        self.assertEqual(len(all_html_files), 4)  # other-file.htmlも含む
        
        # 存在しないパターン
        no_files = self.batch_processor.discover_html_files("nonexistent-*.html")
        self.assertEqual(len(no_files), 0)
    
    def test_is_file_processed(self):
        """処理済みファイル確認のテスト"""
        test_file = os.path.join(self.temp_dir, "page-test.html")
        
        # モックの設定
        self.mock_html_processor.get_file_key.return_value = "page-test"
        
        # 未処理の場合
        self.mock_database_manager.get_vector.return_value = None
        self.assertFalse(self.batch_processor.is_file_processed(test_file))
        
        # 処理済みの場合
        self.mock_database_manager.get_vector.return_value = np.array([1, 2, 3])
        self.assertTrue(self.batch_processor.is_file_processed(test_file))
        
        # エラーが発生した場合
        self.mock_database_manager.get_vector.side_effect = Exception("DB Error")
        self.assertFalse(self.batch_processor.is_file_processed(test_file))
    
    def test_process_single_file_success(self):
        """単一ファイル処理成功のテスト"""
        test_file = os.path.join(self.temp_dir, "page-test.html")
        
        # モックの設定
        self.mock_html_processor.get_file_key.return_value = "page-test"
        self.mock_html_processor.extract_text.return_value = "テストテキスト"
        self.mock_html_processor.validate_content_length.return_value = True
        self.mock_vector_embedder.embed_text.return_value = np.array([0.1, 0.2, 0.3])
        self.mock_database_manager.get_vector.return_value = None  # 未処理
        
        # 処理実行
        result = self.batch_processor.process_single_file(test_file)
        
        # 結果確認
        self.assertTrue(result)
        
        # メソッド呼び出し確認
        self.mock_html_processor.get_file_key.assert_called_with(test_file)
        self.mock_html_processor.extract_text.assert_called_with(test_file)
        self.mock_html_processor.validate_content_length.assert_called_with("テストテキスト")
        self.mock_vector_embedder.embed_text.assert_called_with("テストテキスト")
        self.mock_database_manager.store_vector.assert_called_once()
    
    def test_process_single_file_skip_processed(self):
        """処理済みファイルのスキップテスト"""
        test_file = os.path.join(self.temp_dir, "page-test.html")
        
        # モックの設定（処理済み）
        self.mock_html_processor.get_file_key.return_value = "page-test"
        self.mock_database_manager.get_vector.return_value = np.array([1, 2, 3])
        
        # 処理実行（再処理なし）
        result = self.batch_processor.process_single_file(test_file, force_reprocess=False)
        
        # スキップされることを確認
        self.assertFalse(result)
        self.mock_html_processor.extract_text.assert_not_called()
    
    def test_process_single_file_force_reprocess(self):
        """強制再処理のテスト"""
        test_file = os.path.join(self.temp_dir, "page-test.html")
        
        # モックの設定（処理済みだが強制再処理）
        self.mock_html_processor.get_file_key.return_value = "page-test"
        self.mock_html_processor.extract_text.return_value = "テストテキスト"
        self.mock_html_processor.validate_content_length.return_value = True
        self.mock_vector_embedder.embed_text.return_value = np.array([0.1, 0.2, 0.3])
        self.mock_database_manager.get_vector.return_value = np.array([1, 2, 3])  # 処理済み
        
        # 強制再処理実行
        result = self.batch_processor.process_single_file(test_file, force_reprocess=True)
        
        # 処理されることを確認
        self.assertTrue(result)
        self.mock_html_processor.extract_text.assert_called_with(test_file)
    
    def test_process_single_file_error(self):
        """単一ファイル処理エラーのテスト"""
        test_file = os.path.join(self.temp_dir, "page-test.html")
        
        # モックの設定（エラー発生）
        self.mock_html_processor.get_file_key.return_value = "page-test"
        self.mock_html_processor.extract_text.side_effect = Exception("HTML処理エラー")
        self.mock_database_manager.get_vector.return_value = None
        
        # エラーが発生することを確認
        with self.assertRaises(RuntimeError):
            self.batch_processor.process_single_file(test_file)
    
    def test_display_progress(self):
        """プログレス表示のテスト"""
        import time
        start_time = time.time()
        
        # ログ出力をキャプチャするためのモック
        with patch.object(self.batch_processor.logger, 'info') as mock_info:
            # プログレス表示実行
            self.batch_processor.display_progress(5, 10, "test.html", start_time)
            
            # ログが呼び出されたことを確認
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            
            # 進捗情報が含まれていることを確認
            self.assertIn("進捗: 5/10", call_args)
            self.assertIn("50.0%", call_args)
            self.assertIn("test.html", call_args)
    
    def test_process_batch_success(self):
        """バッチ処理成功のテスト"""
        # モックの設定
        self.mock_html_processor.get_file_key.side_effect = lambda x: os.path.basename(x).replace('.html', '')
        self.mock_html_processor.extract_text.return_value = "テストテキスト"
        self.mock_html_processor.validate_content_length.return_value = True
        self.mock_vector_embedder.embed_text.return_value = np.array([0.1, 0.2, 0.3])
        self.mock_database_manager.get_vector.return_value = None  # 全て未処理
        
        # バッチ処理実行
        result = self.batch_processor.process_batch(show_progress=False)
        
        # 結果確認
        self.assertEqual(result.total_files, 3)  # page-*.htmlファイルが3つ
        self.assertEqual(result.processed_files, 3)
        self.assertEqual(result.skipped_files, 0)
        self.assertEqual(result.error_files, 0)
        self.assertEqual(len(result.errors), 0)
        self.assertGreater(result.processing_time, 0)
    
    def test_process_batch_with_skips_and_errors(self):
        """スキップとエラーを含むバッチ処理のテスト"""
        # モックの設定
        def mock_get_file_key(file_path):
            return os.path.basename(file_path).replace('.html', '')
        
        def mock_get_vector(key):
            # 1つ目のファイルは処理済み
            if "徳川家康" in key:
                return np.array([1, 2, 3])
            return None
        
        def mock_extract_text(file_path):
            # 2つ目のファイルでエラー発生
            if "織田信長" in file_path:
                raise Exception("テストエラー")
            return "テストテキスト"
        
        self.mock_html_processor.get_file_key.side_effect = mock_get_file_key
        self.mock_html_processor.extract_text.side_effect = mock_extract_text
        self.mock_html_processor.validate_content_length.return_value = True
        self.mock_vector_embedder.embed_text.return_value = np.array([0.1, 0.2, 0.3])
        self.mock_database_manager.get_vector.side_effect = mock_get_vector
        
        # バッチ処理実行
        result = self.batch_processor.process_batch(show_progress=False)
        
        # 結果確認
        self.assertEqual(result.total_files, 3)
        self.assertEqual(result.processed_files, 1)  # 豊臣秀吉のみ処理
        self.assertEqual(result.skipped_files, 1)    # 徳川家康はスキップ
        self.assertEqual(result.error_files, 1)      # 織田信長はエラー
        self.assertEqual(len(result.errors), 1)
        
        # エラー詳細確認
        error_file, error_msg = result.errors[0]
        self.assertIn("織田信長", error_file)
        self.assertIn("テストエラー", error_msg)
    
    def test_process_batch_no_files(self):
        """処理対象ファイルなしのテスト"""
        # 存在しないパターンでバッチ処理
        result = self.batch_processor.process_batch(file_pattern="nonexistent-*.html")
        
        # 結果確認
        self.assertEqual(result.total_files, 0)
        self.assertEqual(result.processed_files, 0)
        self.assertEqual(result.skipped_files, 0)
        self.assertEqual(result.error_files, 0)
    
    def test_print_summary(self):
        """サマリー表示のテスト"""
        # テスト用の処理結果を作成
        result = ProcessingResult()
        result.total_files = 10
        result.processed_files = 7
        result.skipped_files = 2
        result.error_files = 1
        result.processing_time = 15.5
        result.errors = [("error_file.html", "テストエラー")]
        
        # ログ出力をキャプチャ
        with patch.object(self.batch_processor.logger, 'info') as mock_info, \
             patch.object(self.batch_processor.logger, 'error') as mock_error:
            
            self.batch_processor.print_summary(result)
            
            # ログが呼び出されたことを確認
            self.assertTrue(mock_info.called)
            self.assertTrue(mock_error.called)
            
            # サマリー情報が含まれていることを確認
            info_calls = [call[0][0] for call in mock_info.call_args_list]
            summary_text = " ".join(info_calls)
            
            self.assertIn("総ファイル数:     10", summary_text)
            self.assertIn("処理済みファイル: 7", summary_text)
            self.assertIn("スキップファイル: 2", summary_text)
            self.assertIn("エラーファイル:   1", summary_text)
            self.assertIn("処理時間:         15.50秒", summary_text)
    
    def test_get_processing_statistics(self):
        """統計情報取得のテスト"""
        # モックの設定
        self.mock_database_manager.get_vector_count.return_value = 5
        self.mock_database_manager.get_all_keys.return_value = ["key1", "key2", "key3", "key4", "key5"]
        self.mock_database_manager.db_path = "test.db"
        
        # 統計情報取得
        stats = self.batch_processor.get_processing_statistics()
        
        # 結果確認
        self.assertEqual(stats["total_vectors"], 5)
        self.assertEqual(stats["total_keys"], 5)
        self.assertEqual(stats["database_path"], "test.db")
    
    def test_get_processing_statistics_error(self):
        """統計情報取得エラーのテスト"""
        # モックの設定（エラー発生）
        self.mock_database_manager.get_vector_count.side_effect = Exception("DB Error")
        
        # 統計情報取得
        stats = self.batch_processor.get_processing_statistics()
        
        # エラー情報が含まれることを確認
        self.assertIn("error", stats)
        self.assertEqual(stats["error"], "DB Error")


class TestProcessingResult(unittest.TestCase):
    """ProcessingResultクラスのテストケース"""
    
    def test_init_default(self):
        """デフォルト初期化のテスト"""
        result = ProcessingResult()
        
        self.assertEqual(result.total_files, 0)
        self.assertEqual(result.processed_files, 0)
        self.assertEqual(result.skipped_files, 0)
        self.assertEqual(result.error_files, 0)
        self.assertEqual(result.processing_time, 0.0)
        self.assertEqual(result.errors, [])
    
    def test_init_with_values(self):
        """値指定初期化のテスト"""
        errors = [("file1.html", "error1"), ("file2.html", "error2")]
        result = ProcessingResult(
            total_files=10,
            processed_files=8,
            skipped_files=1,
            error_files=1,
            processing_time=25.5,
            errors=errors
        )
        
        self.assertEqual(result.total_files, 10)
        self.assertEqual(result.processed_files, 8)
        self.assertEqual(result.skipped_files, 1)
        self.assertEqual(result.error_files, 1)
        self.assertEqual(result.processing_time, 25.5)
        self.assertEqual(result.errors, errors)


if __name__ == '__main__':
    # ログレベルを設定してテスト実行
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    unittest.main()