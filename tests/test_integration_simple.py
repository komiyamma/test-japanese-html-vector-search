"""
シンプルな統合テスト（依存関係を最小限に抑制）
基本的な統合テストとエンドツーエンドテストを実装
"""

import unittest
import os
import tempfile
import shutil
import sqlite3
import time
import numpy as np
from pathlib import Path
import hashlib
import glob

# BeautifulSoupのみインポート
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


class SimpleHTMLProcessor:
    """シンプルなHTMLプロセッサー"""
    
    def extract_text(self, file_path):
        """HTMLファイルからテキストを抽出"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if BS4_AVAILABLE:
            soup = BeautifulSoup(content, 'html.parser')
            return soup.get_text()
        else:
            # BeautifulSoupが利用できない場合の簡易実装
            import re
            # HTMLタグを除去
            text = re.sub(r'<[^>]+>', '', content)
            # 余分な空白を除去
            text = re.sub(r'\s+', ' ', text).strip()
            return text
    
    def get_file_key(self, file_path):
        """ファイルパスからキーを生成"""
        return os.path.splitext(os.path.basename(file_path))[0]


class SimpleVectorEmbedder:
    """シンプルなベクトル埋め込み（ハッシュベース）"""
    
    def __init__(self, vector_dim=384):
        self.vector_dim = vector_dim
    
    def embed_text(self, text):
        """テキストをベクトルに変換（ハッシュベース）"""
        # テキストのハッシュを使用してシードを生成
        hash_obj = hashlib.md5(text.encode())
        seed = int(hash_obj.hexdigest()[:8], 16)
        
        # シードを使用してランダムベクトルを生成
        np.random.seed(seed)
        vector = np.random.randn(self.vector_dim).astype(np.float32)
        
        # 正規化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector


class SimpleDatabaseManager:
    """シンプルなデータベースマネージャー"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.connection = None
    
    def create_table(self):
        """テーブルを作成"""
        self.connection = sqlite3.connect(self.db_path)
        cursor = self.connection.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_key TEXT UNIQUE NOT NULL,
                vector_data BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.connection.commit()
    
    def store_vector(self, key, vector):
        """ベクトルを保存"""
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)
        
        cursor = self.connection.cursor()
        vector_blob = vector.tobytes()
        
        cursor.execute('''
            INSERT OR REPLACE INTO document_vectors (document_key, vector_data)
            VALUES (?, ?)
        ''', (key, vector_blob))
        self.connection.commit()
    
    def get_vector(self, key):
        """ベクトルを取得"""
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)
        
        cursor = self.connection.cursor()
        cursor.execute('''
            SELECT vector_data FROM document_vectors WHERE document_key = ?
        ''', (key,))
        
        result = cursor.fetchone()
        if result:
            vector_blob = result[0]
            return np.frombuffer(vector_blob, dtype=np.float32)
        return None
    
    def get_all_vectors(self):
        """すべてのベクトルを取得"""
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)
        
        cursor = self.connection.cursor()
        cursor.execute('SELECT document_key, vector_data FROM document_vectors')
        
        vectors = {}
        for row in cursor.fetchall():
            key, vector_blob = row
            vectors[key] = np.frombuffer(vector_blob, dtype=np.float32)
        
        return vectors


class SimpleSimilarityCalculator:
    """シンプルな類似度計算"""
    
    def cosine_similarity(self, v1, v2):
        """コサイン類似度を計算"""
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_similar_documents(self, query_vector, doc_vectors, top_k=5):
        """類似ドキュメントを検索"""
        similarities = []
        
        for key, vector in doc_vectors.items():
            sim = self.cosine_similarity(query_vector, vector)
            similarities.append((key, sim))
        
        # 類似度でソート
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class SimpleBatchProcessor:
    """シンプルなバッチプロセッサー"""
    
    def __init__(self, html_processor, vector_embedder, db_manager):
        self.html_processor = html_processor
        self.vector_embedder = vector_embedder
        self.db_manager = db_manager
    
    def process_directory(self, directory):
        """ディレクトリ内のHTMLファイルを処理"""
        html_files = glob.glob(os.path.join(directory, "page-*.html"))
        processed = 0
        errors = 0
        
        for file_path in html_files:
            try:
                # テキスト抽出
                text = self.html_processor.extract_text(file_path)
                
                # ベクトル化
                vector = self.vector_embedder.embed_text(text)
                
                # データベースに保存
                key = self.html_processor.get_file_key(file_path)
                self.db_manager.store_vector(key, vector)
                
                processed += 1
                
            except Exception as e:
                print(f"エラー: {file_path} - {e}")
                errors += 1
        
        return {"processed": processed, "errors": errors}


class SimpleQueryEngine:
    """シンプルなクエリエンジン"""
    
    def __init__(self, vector_embedder, db_manager, similarity_calculator):
        self.vector_embedder = vector_embedder
        self.db_manager = db_manager
        self.similarity_calculator = similarity_calculator
    
    def search_by_text(self, query_text, top_k=5):
        """テキストクエリで検索"""
        query_vector = self.vector_embedder.embed_text(query_text)
        all_vectors = self.db_manager.get_all_vectors()
        return self.similarity_calculator.find_similar_documents(
            query_vector, all_vectors, top_k
        )
    
    def search_by_document(self, doc_key, top_k=5):
        """ドキュメントキーで類似検索"""
        doc_vector = self.db_manager.get_vector(doc_key)
        if doc_vector is None:
            return []
        
        all_vectors = self.db_manager.get_all_vectors()
        # 自分自身を除外
        filtered_vectors = {k: v for k, v in all_vectors.items() if k != doc_key}
        
        return self.similarity_calculator.find_similar_documents(
            doc_vector, filtered_vectors, top_k
        )


class TestSimpleIntegration(unittest.TestCase):
    """シンプルな統合テストクラス"""
    
    def setUp(self):
        """テスト前の準備"""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_vectors.db")
        
        # テストデータディレクトリのパス
        self.test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
        
        # コンポーネントを初期化
        self.html_processor = SimpleHTMLProcessor()
        self.vector_embedder = SimpleVectorEmbedder()
        self.db_manager = SimpleDatabaseManager(self.db_path)
        self.similarity_calculator = SimpleSimilarityCalculator()
        self.batch_processor = SimpleBatchProcessor(
            self.html_processor,
            self.vector_embedder,
            self.db_manager
        )
        self.query_engine = SimpleQueryEngine(
            self.vector_embedder,
            self.db_manager,
            self.similarity_calculator
        )
        
        # データベーステーブルを作成
        self.db_manager.create_table()
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        if self.db_manager.connection:
            self.db_manager.connection.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_end_to_end_workflow(self):
        """エンドツーエンドワークフローテスト"""
        print("\nエンドツーエンドワークフローテスト")
        
        # 1. HTMLファイルからテキスト抽出
        sample_file = os.path.join(self.test_data_dir, "sample_page_1.html")
        self.assertTrue(os.path.exists(sample_file), "テストファイルが存在しません")
        
        text = self.html_processor.extract_text(sample_file)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)
        print(f"  ✅ テキスト抽出完了: {len(text)}文字")
        
        # 2. テキストをベクトル化
        vector = self.vector_embedder.embed_text(text)
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(len(vector.shape), 1)
        self.assertEqual(vector.shape[0], 384)
        print(f"  ✅ ベクトル化完了: {vector.shape}")
        
        # 3. ベクトルをデータベースに保存
        key = self.html_processor.get_file_key(sample_file)
        self.db_manager.store_vector(key, vector)
        print(f"  ✅ データベース保存完了: {key}")
        
        # 4. データベースからベクトルを取得
        retrieved_vector = self.db_manager.get_vector(key)
        self.assertIsNotNone(retrieved_vector)
        np.testing.assert_array_almost_equal(vector, retrieved_vector)
        print("  ✅ データベース取得確認")
        
        # 5. 類似度検索を実行
        query_text = "戦国時代の武将"
        results = self.query_engine.search_by_text(query_text, top_k=1)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0][0], key)
        self.assertIsInstance(results[0][1], (float, np.floating))
        print(f"  ✅ 類似度検索完了: {results[0][1]:.3f}")
        
        print("  エンドツーエンドワークフロー成功！")
    
    def test_batch_processing_integration(self):
        """バッチ処理の統合テスト"""
        print("\nバッチ処理統合テスト")
        
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
        
        print(f"  ✅ テストファイル準備: {len(test_files)}個")
        
        # バッチ処理を実行
        results = self.batch_processor.process_directory(temp_html_dir)
        
        # 結果を検証
        self.assertIsInstance(results, dict)
        self.assertIn("processed", results)
        self.assertIn("errors", results)
        self.assertEqual(results["processed"], 3)
        self.assertEqual(results["errors"], 0)
        print(f"  ✅ バッチ処理完了: {results['processed']}個処理")
        
        # データベースに保存されたことを確認
        all_vectors = self.db_manager.get_all_vectors()
        self.assertEqual(len(all_vectors), 3)
        print(f"  ✅ データベース確認: {len(all_vectors)}個保存")
        
        print("  バッチ処理統合テスト成功！")
    
    def test_similarity_search_integration(self):
        """類似度検索の統合テスト"""
        print("\n類似度検索統合テスト")
        
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
        
        print(f"  ✅ ドキュメント準備: {len(test_files)}個")
        
        # テキストクエリによる検索
        results = self.query_engine.search_by_text("天下統一", top_k=3)
        self.assertEqual(len(results), 3)
        
        # 結果がスコア順にソートされていることを確認
        scores = [result[1] for result in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        print(f"  ✅ テキスト検索完了: {len(results)}件")
        
        # ドキュメントキーによる検索
        similar_docs = self.query_engine.search_by_document("page-bushou-織田信長", top_k=2)
        self.assertEqual(len(similar_docs), 2)
        
        # 自分自身は結果に含まれないことを確認
        result_keys = [result[0] for result in similar_docs]
        self.assertNotIn("page-bushou-織田信長", result_keys)
        print(f"  ✅ ドキュメント検索完了: {len(similar_docs)}件")
        
        print("  類似度検索統合テスト成功！")
    
    def test_performance_basic(self):
        """基本的なパフォーマンステスト"""
        print("\n基本パフォーマンステスト")
        
        # 大きなテキストファイルを処理
        large_file = os.path.join(self.test_data_dir, "large_page.html")
        
        start_time = time.time()
        
        # テキスト抽出
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
        
        print(f"  ✅ テキスト抽出: {extract_time:.3f}秒")
        print(f"  ✅ ベクトル化: {vector_time:.3f}秒")
        print(f"  ✅ DB保存: {db_time:.3f}秒")
        print(f"  ✅ 総処理時間: {total_time:.3f}秒")
        
        # 基本的なパフォーマンス要件
        self.assertLess(total_time, 5.0, "総処理時間が5秒以内であること")
        
        print("  基本パフォーマンステスト成功！")
    
    def test_error_handling(self):
        """エラーハンドリングテスト"""
        print("\nエラーハンドリングテスト")
        
        # 存在しないファイル
        with self.assertRaises(FileNotFoundError):
            self.html_processor.extract_text("nonexistent.html")
        print("  ✅ 存在しないファイルのエラーハンドリング")
        
        # 空のテキスト
        empty_vector = self.vector_embedder.embed_text("")
        self.assertIsInstance(empty_vector, np.ndarray)
        print("  ✅ 空テキストの処理")
        
        # 存在しないキーの取得
        result = self.db_manager.get_vector("nonexistent_key")
        self.assertIsNone(result)
        print("  ✅ 存在しないキーの処理")
        
        print("  エラーハンドリングテスト成功！")


if __name__ == '__main__':
    print("シンプル統合テスト開始")
    print("=" * 50)
    
    # BeautifulSoupの利用可能性を確認
    if BS4_AVAILABLE:
        print("✅ BeautifulSoup4が利用可能")
    else:
        print("⚠️ BeautifulSoup4が利用できません（簡易HTMLパーサーを使用）")
    
    # テストを実行
    unittest.main(verbosity=2)