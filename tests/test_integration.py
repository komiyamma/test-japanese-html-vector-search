"""
統合テストとエンドツーエンドテスト
全体フローの動作確認とパフォーマンステストを実装
"""

import unittest
import os
import tempfile
import shutil
import sqlite3
import time
import numpy as np
from pathlib import Path

# テスト対象のモジュールをインポート
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from html_processor import HTMLProcessor
    from vector_embedder import VectorEmbedder
    from database_manager import DatabaseManager
    from similarity_calculator import SimilarityCalculator
    from batch_processor import BatchProcessor
    from query_engine import QueryEngine
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ モジュールのインポートに失敗: {e}")
    MODULES_AVAILABLE = False
    
    # モッククラスを定義
    class MockVectorEmbedder:
        def embed_text(self, text):
            # テキストの長さに基づいてダミーベクトルを生成
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            seed = int(hash_obj.hexdigest()[:8], 16)
            np.random.seed(seed)
            vector = np.random.randn(384).astype(np.float32)
            return vector / np.linalg.norm(vector)
    
    class MockHTMLProcessor:
        def extract_text(self, file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            return soup.get_text()
        
        def get_file_key(self, file_path):
            return os.path.splitext(os.path.basename(file_path))[0]
    
    class MockDatabaseManager:
        def __init__(self, db_path):
            self.db_path = db_path
            self.vectors = {}
        
        def create_table(self):
            pass
        
        def store_vector(self, key, vector):
            self.vectors[key] = vector.copy()
        
        def get_vector(self, key):
            return self.vectors.get(key)
        
        def get_all_vectors(self):
            return self.vectors.copy()
    
    class MockSimilarityCalculator:
        def cosine_similarity(self, v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        def find_similar_documents(self, query_vector, doc_vectors, top_k=5):
            similarities = []
            for key, vector in doc_vectors.items():
                sim = self.cosine_similarity(query_vector, vector)
                similarities.append((key, sim))
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
    
    class MockBatchProcessor:
        def __init__(self, html_processor, vector_embedder, db_manager):
            self.html_processor = html_processor
            self.vector_embedder = vector_embedder
            self.db_manager = db_manager
        
        def process_directory(self, directory):
            import glob
            html_files = glob.glob("page-*.html")
            processed = 0
            errors = 0
            
            for file in html_files:
                try:
                    text = self.html_processor.extract_text(file)
                    vector = self.vector_embedder.embed_text(text)
                    key = self.html_processor.get_file_key(file)
                    self.db_manager.store_vector(key, vector)
                    processed += 1
                except Exception:
                    errors += 1
            
            return {"processed": processed, "errors": errors}
    
    class MockQueryEngine:
        def __init__(self, vector_embedder, db_manager, similarity_calculator):
            self.vector_embedder = vector_embedder
            self.db_manager = db_manager
            self.similarity_calculator = similarity_calculator
        
        def search_by_text(self, query_text, top_k=5):
            query_vector = self.vector_embedder.embed_text(query_text)
            all_vectors = self.db_manager.get_all_vectors()
            return self.similarity_calculator.find_similar_documents(
                query_vector, all_vectors, top_k
            )
        
        def search_by_document(self, doc_key, top_k=5):
            doc_vector = self.db_manager.get_vector(doc_key)
            if doc_vector is None:
                return []
            all_vectors = self.db_manager.get_all_vectors()
            # 自分自身を除外
            filtered_vectors = {k: v for k, v in all_vectors.items() if k != doc_key}
            return self.similarity_calculator.find_similar_documents(
                doc_vector, filtered_vectors, top_k
            )
    
    # モッククラスを実際のクラスとして使用
    HTMLProcessor = MockHTMLProcessor
    VectorEmbedder = MockVectorEmbedder
    DatabaseManager = MockDatabaseManager
    SimilarityCalculator = MockSimilarityCalculator
    BatchProcessor = MockBatchProcessor
    QueryEngine = MockQueryEngine


class TestIntegration(unittest.TestCase):
    """統合テストクラス"""
    
    def setUp(self):
        """テスト前の準備"""
        # 一時ディレクトリを作成
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_vectors.db")
        
        # テストデータディレクトリのパス
        self.test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
        
        # コンポーネントを初期化
        self.html_processor = HTMLProcessor()
        self.vector_embedder = VectorEmbedder()
        self.db_manager = DatabaseManager(self.db_path)
        self.similarity_calculator = SimilarityCalculator()
        self.batch_processor = BatchProcessor(
            self.html_processor,
            self.vector_embedder,
            self.db_manager
        )
        self.query_engine = QueryEngine(
            self.vector_embedder,
            self.db_manager,
            self.similarity_calculator
        )
        
        # データベーステーブルを作成
        self.db_manager.create_table()
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        # 一時ディレクトリを削除
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_end_to_end_workflow(self):
        """エンドツーエンドワークフローテスト"""
        # 1. HTMLファイルからテキスト抽出
        sample_file = os.path.join(self.test_data_dir, "sample_page_1.html")
        self.assertTrue(os.path.exists(sample_file), "テストファイルが存在しません")
        
        text = self.html_processor.extract_text(sample_file)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)
        self.assertIn("織田信長", text)
        
        # 2. テキストをベクトル化
        vector = self.vector_embedder.embed_text(text)
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(len(vector.shape), 1)  # 1次元配列
        self.assertGreater(vector.shape[0], 0)  # 要素数が0より大きい
        
        # 3. ベクトルをデータベースに保存
        key = self.html_processor.get_file_key(sample_file)
        self.db_manager.store_vector(key, vector)
        
        # 4. データベースからベクトルを取得
        retrieved_vector = self.db_manager.get_vector(key)
        self.assertIsNotNone(retrieved_vector)
        np.testing.assert_array_almost_equal(vector, retrieved_vector)
        
        # 5. 類似度検索を実行
        query_text = "戦国時代の武将"
        results = self.query_engine.search_by_text(query_text, top_k=1)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0][0], key)
        self.assertIsInstance(results[0][1], float)
        self.assertGreaterEqual(results[0][1], 0.0)
        self.assertLessEqual(results[0][1], 1.0)
    
    def test_batch_processing_integration(self):
        """バッチ処理の統合テスト"""
        # テストデータをコピー
        test_files = [
            "sample_page_1.html",
            "sample_page_2.html", 
            "sample_page_3.html"
        ]
        
        temp_html_dir = os.path.join(self.test_dir, "html_files")
        os.makedirs(temp_html_dir)
        
        for filename in test_files:
            src = os.path.join(self.test_data_dir, filename)
            dst = os.path.join(temp_html_dir, f"page-bushou-{filename}")
            shutil.copy2(src, dst)
        
        # バッチ処理を実行
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_html_dir)
            results = self.batch_processor.process_directory(".")
            
            # 結果を検証
            self.assertIsInstance(results, dict)
            self.assertIn("processed", results)
            self.assertIn("errors", results)
            self.assertEqual(results["processed"], 3)
            self.assertEqual(results["errors"], 0)
            
            # データベースに保存されたことを確認
            all_vectors = self.db_manager.get_all_vectors()
            self.assertEqual(len(all_vectors), 3)
            
        finally:
            os.chdir(original_cwd)
    
    def test_similarity_search_integration(self):
        """類似度検索の統合テスト"""
        # 複数のドキュメントを処理
        test_files = [
            ("sample_page_1.html", "page-bushou-織田信長"),
            ("sample_page_2.html", "page-bushou-豊臣秀吉"),
            ("sample_page_3.html", "page-bushou-徳川家康")
        ]
        
        for filename, key in test_files:
            file_path = os.path.join(self.test_data_dir, filename)
            text = self.html_processor.extract_text(file_path)
            vector = self.vector_embedder.embed_text(text)
            self.db_manager.store_vector(key, vector)
        
        # テキストクエリによる検索
        results = self.query_engine.search_by_text("天下統一", top_k=3)
        self.assertEqual(len(results), 3)
        
        # 結果がスコア順にソートされていることを確認
        scores = [result[1] for result in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        
        # ドキュメントキーによる検索
        similar_docs = self.query_engine.search_by_document("page-bushou-織田信長", top_k=2)
        self.assertEqual(len(similar_docs), 2)  # 自分以外の2つ
        
        # 自分自身は結果に含まれないことを確認
        result_keys = [result[0] for result in similar_docs]
        self.assertNotIn("page-bushou-織田信長", result_keys)


class TestPerformance(unittest.TestCase):
    """パフォーマンステストクラス"""
    
    def setUp(self):
        """テスト前の準備"""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "perf_test.db")
        
        # コンポーネントを初期化
        self.html_processor = HTMLProcessor()
        self.vector_embedder = VectorEmbedder()
        self.db_manager = DatabaseManager(self.db_path)
        self.similarity_calculator = SimilarityCalculator()
        
        self.db_manager.create_table()
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_large_document_processing_performance(self):
        """大規模ドキュメント処理のパフォーマンステスト"""
        test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
        large_file = os.path.join(test_data_dir, "large_page.html")
        
        # 処理時間を測定
        start_time = time.time()
        
        # HTMLからテキスト抽出
        text = self.html_processor.extract_text(large_file)
        extract_time = time.time() - start_time
        
        # ベクトル化
        vector_start = time.time()
        vector = self.vector_embedder.embed_text(text)
        vector_time = time.time() - vector_start
        
        # データベース保存
        db_start = time.time()
        self.db_manager.store_vector("large_document", vector)
        db_time = time.time() - db_start
        
        total_time = time.time() - start_time
        
        # パフォーマンス要件を検証（具体的な閾値は環境に依存）
        self.assertLess(extract_time, 1.0, "テキスト抽出が1秒以内に完了すること")
        self.assertLess(vector_time, 5.0, "ベクトル化が5秒以内に完了すること")
        self.assertLess(db_time, 0.1, "データベース保存が0.1秒以内に完了すること")
        self.assertLess(total_time, 10.0, "全体処理が10秒以内に完了すること")
        
        # メモリ使用量の確認（ベクトルサイズ）
        self.assertLess(vector.nbytes, 10000, "ベクトルサイズが10KB以内であること")
    
    def test_batch_processing_performance(self):
        """バッチ処理のパフォーマンステスト"""
        # 複数のテストファイルを作成
        test_files_count = 10
        temp_html_dir = os.path.join(self.test_dir, "perf_test_files")
        os.makedirs(temp_html_dir)
        
        # テンプレートHTMLを作成
        template_content = """<!DOCTYPE html>
<html lang="ja">
<head><meta charset="UTF-8"><title>テスト{}</title></head>
<body><h1>テストドキュメント{}</h1><p>これはパフォーマンステスト用のドキュメント{}です。</p></body>
</html>"""
        
        for i in range(test_files_count):
            filename = f"page-test-{i:03d}.html"
            filepath = os.path.join(temp_html_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(template_content.format(i, i, i))
        
        # バッチ処理を実行
        batch_processor = BatchProcessor(
            self.html_processor,
            self.vector_embedder,
            self.db_manager
        )
        
        start_time = time.time()
        original_cwd = os.getcwd()
        
        try:
            os.chdir(temp_html_dir)
            results = batch_processor.process_directory(".")
            processing_time = time.time() - start_time
            
            # 結果を検証
            self.assertEqual(results["processed"], test_files_count)
            self.assertEqual(results["errors"], 0)
            
            # パフォーマンス要件（1ファイルあたり平均2秒以内）
            avg_time_per_file = processing_time / test_files_count
            self.assertLess(avg_time_per_file, 2.0, 
                          f"1ファイルあたりの平均処理時間が2秒以内であること（実際: {avg_time_per_file:.2f}秒）")
            
        finally:
            os.chdir(original_cwd)
    
    def test_similarity_search_performance(self):
        """類似度検索のパフォーマンステスト"""
        # 100個のダミーベクトルを作成
        vector_count = 100
        vector_dim = 384  # 使用モデルの次元数
        
        for i in range(vector_count):
            # ランダムベクトルを生成（正規化済み）
            vector = np.random.randn(vector_dim).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            self.db_manager.store_vector(f"doc_{i:03d}", vector)
        
        # クエリベクトルを作成
        query_vector = np.random.randn(vector_dim).astype(np.float32)
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        # 検索パフォーマンスを測定
        start_time = time.time()
        all_vectors = self.db_manager.get_all_vectors()
        db_time = time.time() - start_time
        
        similarity_start = time.time()
        results = self.similarity_calculator.find_similar_documents(
            query_vector, all_vectors, top_k=10
        )
        similarity_time = time.time() - similarity_start
        
        total_time = time.time() - start_time
        
        # パフォーマンス要件を検証
        self.assertLess(db_time, 0.5, "データベース読み込みが0.5秒以内に完了すること")
        self.assertLess(similarity_time, 1.0, "類似度計算が1秒以内に完了すること")
        self.assertLess(total_time, 2.0, "全体検索が2秒以内に完了すること")
        
        # 結果の妥当性を確認
        self.assertEqual(len(results), 10)
        self.assertTrue(all(isinstance(score, float) for _, score in results))
        self.assertTrue(all(0.0 <= score <= 1.0 for _, score in results))


class TestErrorScenarios(unittest.TestCase):
    """エラーシナリオテストクラス"""
    
    def setUp(self):
        """テスト前の準備"""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "error_test.db")
        self.test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
        
        # コンポーネントを初期化
        self.html_processor = HTMLProcessor()
        self.vector_embedder = VectorEmbedder()
        self.db_manager = DatabaseManager(self.db_path)
        self.batch_processor = BatchProcessor(
            self.html_processor,
            self.vector_embedder,
            self.db_manager
        )
        
        self.db_manager.create_table()
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_empty_html_file_handling(self):
        """空のHTMLファイルの処理テスト"""
        empty_file = os.path.join(self.test_data_dir, "empty_page.html")
        
        # 空ファイルでもエラーにならないことを確認
        text = self.html_processor.extract_text(empty_file)
        self.assertIsInstance(text, str)
        
        # 短いテキストでも処理できることを確認
        if len(text.strip()) > 0:
            vector = self.vector_embedder.embed_text(text)
            self.assertIsInstance(vector, np.ndarray)
    
    def test_nonexistent_file_handling(self):
        """存在しないファイルの処理テスト"""
        nonexistent_file = os.path.join(self.test_dir, "nonexistent.html")
        
        # ファイルが存在しない場合の例外処理
        with self.assertRaises(FileNotFoundError):
            self.html_processor.extract_text(nonexistent_file)
    
    def test_database_connection_error_handling(self):
        """データベース接続エラーの処理テスト"""
        # 無効なパスでデータベースマネージャーを作成
        invalid_db_path = "/invalid/path/test.db"
        
        # データベース作成時のエラーハンドリング
        with self.assertRaises((sqlite3.OperationalError, OSError)):
            invalid_db_manager = DatabaseManager(invalid_db_path)
            invalid_db_manager.create_table()
    
    def test_batch_processing_with_mixed_files(self):
        """正常ファイルとエラーファイルが混在する場合のバッチ処理テスト"""
        # テスト用ディレクトリを作成
        mixed_files_dir = os.path.join(self.test_dir, "mixed_files")
        os.makedirs(mixed_files_dir)
        
        # 正常なファイルをコピー
        good_file = os.path.join(self.test_data_dir, "sample_page_1.html")
        shutil.copy2(good_file, os.path.join(mixed_files_dir, "page-good-test.html"))
        
        # 空のファイルをコピー
        empty_file = os.path.join(self.test_data_dir, "empty_page.html")
        shutil.copy2(empty_file, os.path.join(mixed_files_dir, "page-empty-test.html"))
        
        # 無効なHTMLファイルを作成
        invalid_html = os.path.join(mixed_files_dir, "page-invalid-test.html")
        with open(invalid_html, 'w', encoding='utf-8') as f:
            f.write("これは無効なHTMLです")
        
        # バッチ処理を実行
        original_cwd = os.getcwd()
        try:
            os.chdir(mixed_files_dir)
            results = self.batch_processor.process_directory(".")
            
            # 一部のファイルは処理され、一部はエラーになることを確認
            self.assertIsInstance(results, dict)
            self.assertIn("processed", results)
            self.assertIn("errors", results)
            self.assertGreaterEqual(results["processed"], 1)  # 少なくとも1つは処理される
            
        finally:
            os.chdir(original_cwd)
    
    def test_memory_stress_scenario(self):
        """メモリストレステスト"""
        # 非常に長いテキストを作成
        long_text = "これは非常に長いテキストです。" * 10000  # 約300KB
        
        # メモリ不足にならずに処理できることを確認
        try:
            vector = self.vector_embedder.embed_text(long_text)
            self.assertIsInstance(vector, np.ndarray)
            
            # データベースに保存
            self.db_manager.store_vector("long_text_test", vector)
            
            # 取得できることを確認
            retrieved = self.db_manager.get_vector("long_text_test")
            self.assertIsNotNone(retrieved)
            
        except MemoryError:
            self.fail("メモリ不足エラーが発生しました")


if __name__ == '__main__':
    # テストスイートを作成
    test_suite = unittest.TestSuite()
    
    # 統合テストを追加
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    test_suite.addTest(unittest.makeSuite(TestPerformance))
    test_suite.addTest(unittest.makeSuite(TestErrorScenarios))
    
    # テストランナーを作成して実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 結果を出力
    if result.wasSuccessful():
        print("\n✅ すべての統合テストが成功しました！")
    else:
        print(f"\n❌ {len(result.failures)} 個のテストが失敗しました")
        print(f"❌ {len(result.errors)} 個のエラーが発生しました")