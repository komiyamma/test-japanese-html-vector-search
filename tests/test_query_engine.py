"""
QueryEngineクラスの単体テスト

このモジュールは、QueryEngineクラスの各機能をテストします。
"""

import unittest
import tempfile
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import logging
import sys

# sentence_transformersのインポートをモック化
sys.modules['sentence_transformers'] = Mock()
sys.modules['torch'] = Mock()

# テスト対象のインポート
from src.query_engine import QueryEngine


class TestQueryEngine(unittest.TestCase):
    """QueryEngineクラスのテストケース"""
    
    def setUp(self):
        """各テストの前に実行される初期化処理"""
        # テスト用の一時データベースファイル
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        # ログレベルを設定（テスト中のログ出力を制御）
        logging.getLogger().setLevel(logging.WARNING)
        
        # テスト用のサンプルベクトル
        self.sample_vector_1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.sample_vector_2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.sample_vector_3 = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        
        # サンプルドキュメントベクトル
        self.sample_documents = {
            "doc1": self.sample_vector_1,
            "doc2": self.sample_vector_2,
            "doc3": self.sample_vector_3
        }
    
    def tearDown(self):
        """各テストの後に実行されるクリーンアップ処理"""
        # 一時ファイルを削除
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    @patch('src.query_engine.VectorEmbedder')
    @patch('src.query_engine.DatabaseManager')
    @patch('src.query_engine.SimilarityCalculator')
    def test_init_success(self, mock_similarity, mock_db, mock_embedder):
        """QueryEngine初期化の正常ケースをテスト"""
        # モックの設定
        mock_embedder.return_value = Mock()
        mock_db.return_value = Mock()
        mock_similarity.return_value = Mock()
        
        # QueryEngineを初期化
        engine = QueryEngine(db_path=self.db_path)
        
        # 初期化が正常に完了することを確認
        self.assertIsNotNone(engine.vector_embedder)
        self.assertIsNotNone(engine.database_manager)
        self.assertIsNotNone(engine.similarity_calculator)
        self.assertEqual(engine.db_path, self.db_path)
    
    @patch('src.query_engine.VectorEmbedder')
    def test_init_failure(self, mock_embedder):
        """QueryEngine初期化の失敗ケースをテスト"""
        # VectorEmbedderの初期化でエラーを発生させる
        mock_embedder.side_effect = RuntimeError("モデル読み込み失敗")
        
        # RuntimeErrorが発生することを確認
        with self.assertRaises(RuntimeError) as context:
            QueryEngine(db_path=self.db_path)
        
        self.assertIn("QueryEngine初期化に失敗", str(context.exception))
    
    def test_search_by_text_success(self):
        """テキスト検索の正常ケースをテスト"""
        with patch.object(QueryEngine, '__init__', lambda x, **kwargs: None):
            engine = QueryEngine()
            
            # モックコンポーネントを設定
            engine.vector_embedder = Mock()
            engine.database_manager = Mock()
            engine.similarity_calculator = Mock()
            
            # モックの戻り値を設定
            engine.vector_embedder.embed_text.return_value = self.sample_vector_1
            engine.database_manager.get_all_vectors.return_value = self.sample_documents
            engine.similarity_calculator.find_similar_documents.return_value = [
                ("doc2", 0.8),
                ("doc3", 0.6)
            ]
            
            # テスト実行
            result = engine.search_by_text("テストクエリ", top_k=2, threshold=0.5)
            
            # 結果を検証
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0], ("doc2", 0.8))
            self.assertEqual(result[1], ("doc3", 0.6))
            
            # メソッドが正しく呼ばれたことを確認
            engine.vector_embedder.embed_text.assert_called_once_with("テストクエリ")
            engine.database_manager.get_all_vectors.assert_called_once()
            engine.similarity_calculator.find_similar_documents.assert_called_once()
    
    def test_search_by_text_empty_query(self):
        """空のクエリテキストでのエラーテスト"""
        with patch.object(QueryEngine, '__init__', lambda x, **kwargs: None):
            engine = QueryEngine()
            
            # 空文字列でValueErrorが発生することを確認
            with self.assertRaises(ValueError) as context:
                engine.search_by_text("")
            
            self.assertIn("検索クエリテキストが空です", str(context.exception))
    
    def test_search_by_text_invalid_parameters(self):
        """無効なパラメータでのエラーテスト"""
        with patch.object(QueryEngine, '__init__', lambda x, **kwargs: None):
            engine = QueryEngine()
            
            # 無効なtop_k
            with self.assertRaises(ValueError) as context:
                engine.search_by_text("テスト", top_k=0)
            self.assertIn("top_kは正の整数", str(context.exception))
            
            # 無効なthreshold
            with self.assertRaises(ValueError) as context:
                engine.search_by_text("テスト", threshold=1.5)
            self.assertIn("thresholdは0.0から1.0の間", str(context.exception))
    
    def test_search_by_text_database_error(self):
        """データベースエラーのテスト"""
        with patch.object(QueryEngine, '__init__', lambda x, **kwargs: None):
            engine = QueryEngine()
            
            # モックコンポーネントを設定
            engine.vector_embedder = Mock()
            engine.database_manager = Mock()
            engine.similarity_calculator = Mock()
            
            # ベクトル化は成功するが、データベース取得でエラー
            engine.vector_embedder.embed_text.return_value = self.sample_vector_1
            engine.database_manager.get_all_vectors.side_effect = Exception("DB接続エラー")
            
            # RuntimeErrorが発生することを確認
            with self.assertRaises(RuntimeError) as context:
                engine.search_by_text("テストクエリ")
            
            self.assertIn("データベースからのベクトル取得に失敗", str(context.exception))
    
    def test_search_by_text_empty_database(self):
        """空のデータベースでのテスト"""
        with patch.object(QueryEngine, '__init__', lambda x, **kwargs: None):
            engine = QueryEngine()
            
            # モックコンポーネントを設定
            engine.vector_embedder = Mock()
            engine.database_manager = Mock()
            engine.similarity_calculator = Mock()
            
            # 空のデータベース
            engine.vector_embedder.embed_text.return_value = self.sample_vector_1
            engine.database_manager.get_all_vectors.return_value = {}
            
            # 空のリストが返されることを確認
            result = engine.search_by_text("テストクエリ")
            self.assertEqual(result, [])
    
    def test_search_by_document_key_success(self):
        """ドキュメントキー検索の正常ケースをテスト"""
        with patch.object(QueryEngine, '__init__', lambda x, **kwargs: None):
            engine = QueryEngine()
            
            # モックコンポーネントを設定
            engine.database_manager = Mock()
            engine.similarity_calculator = Mock()
            
            # モックの戻り値を設定
            engine.database_manager.get_vector.return_value = self.sample_vector_1
            engine.database_manager.get_all_vectors.return_value = self.sample_documents
            engine.similarity_calculator.find_similar_documents.return_value = [
                ("doc2", 0.7),
                ("doc3", 0.5)
            ]
            
            # テスト実行
            result = engine.search_by_document_key("doc1", top_k=2, threshold=0.4)
            
            # 結果を検証
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0], ("doc2", 0.7))
            self.assertEqual(result[1], ("doc3", 0.5))
            
            # メソッドが正しく呼ばれたことを確認
            engine.database_manager.get_vector.assert_called_once_with("doc1")
            engine.database_manager.get_all_vectors.assert_called_once()
    
    def test_search_by_document_key_not_found(self):
        """存在しないドキュメントキーでのエラーテスト"""
        with patch.object(QueryEngine, '__init__', lambda x, **kwargs: None):
            engine = QueryEngine()
            
            # モックコンポーネントを設定
            engine.database_manager = Mock()
            
            # ドキュメントが見つからない場合
            engine.database_manager.get_vector.return_value = None
            
            # ValueErrorが発生することを確認
            with self.assertRaises(ValueError) as context:
                engine.search_by_document_key("nonexistent")
            
            self.assertIn("指定されたドキュメントキーが見つかりません", str(context.exception))
    
    def test_search_by_document_key_exclude_self(self):
        """自分自身を除外する機能のテスト"""
        with patch.object(QueryEngine, '__init__', lambda x, **kwargs: None):
            engine = QueryEngine()
            
            # モックコンポーネントを設定
            engine.database_manager = Mock()
            engine.similarity_calculator = Mock()
            
            # モックの戻り値を設定
            engine.database_manager.get_vector.return_value = self.sample_vector_1
            engine.database_manager.get_all_vectors.return_value = self.sample_documents.copy()
            engine.similarity_calculator.find_similar_documents.return_value = [
                ("doc2", 0.7),
                ("doc3", 0.5)
            ]
            
            # exclude_self=Trueでテスト実行
            result = engine.search_by_document_key("doc1", exclude_self=True)
            
            # similarity_calculatorに渡されたdocument_vectorsにdoc1が含まれていないことを確認
            call_args = engine.similarity_calculator.find_similar_documents.call_args
            passed_vectors = call_args[1]['document_vectors']  # キーワード引数
            self.assertNotIn("doc1", passed_vectors)
            self.assertIn("doc2", passed_vectors)
            self.assertIn("doc3", passed_vectors)
    
    def test_get_document_info_exists(self):
        """存在するドキュメントの情報取得テスト"""
        with patch.object(QueryEngine, '__init__', lambda x, **kwargs: None):
            engine = QueryEngine()
            
            # モックコンポーネントを設定
            engine.database_manager = Mock()
            engine.database_manager.get_vector.return_value = self.sample_vector_1
            
            # テスト実行
            result = engine.get_document_info("doc1")
            
            # 結果を検証
            self.assertEqual(result["key"], "doc1")
            self.assertTrue(result["exists"])
            self.assertEqual(result["vector_dimension"], 3)
    
    def test_get_document_info_not_exists(self):
        """存在しないドキュメントの情報取得テスト"""
        with patch.object(QueryEngine, '__init__', lambda x, **kwargs: None):
            engine = QueryEngine()
            
            # モックコンポーネントを設定
            engine.database_manager = Mock()
            engine.database_manager.get_vector.return_value = None
            
            # テスト実行
            result = engine.get_document_info("nonexistent")
            
            # 結果を検証
            self.assertEqual(result["key"], "nonexistent")
            self.assertFalse(result["exists"])
            self.assertIsNone(result["vector_dimension"])
    
    def test_get_database_stats(self):
        """データベース統計情報取得のテスト"""
        with patch.object(QueryEngine, '__init__', lambda x, **kwargs: None):
            engine = QueryEngine()
            engine.db_path = self.db_path
            engine.model_name = "test-model"
            
            # モックコンポーネントを設定
            engine.database_manager = Mock()
            engine.vector_embedder = Mock()
            
            engine.database_manager.get_vector_count.return_value = 10
            engine.vector_embedder.get_embedding_dimension.return_value = 384
            
            # テスト実行
            result = engine.get_database_stats()
            
            # 結果を検証
            self.assertEqual(result["total_documents"], 10)
            self.assertEqual(result["database_path"], self.db_path)
            self.assertEqual(result["model_name"], "test-model")
            self.assertEqual(result["vector_dimension"], 384)
    
    def test_list_all_documents(self):
        """全ドキュメントキー一覧取得のテスト"""
        with patch.object(QueryEngine, '__init__', lambda x, **kwargs: None):
            engine = QueryEngine()
            
            # モックコンポーネントを設定
            engine.database_manager = Mock()
            engine.database_manager.get_all_keys.return_value = ["doc1", "doc2", "doc3"]
            
            # テスト実行
            result = engine.list_all_documents()
            
            # 結果を検証
            self.assertEqual(result, ["doc1", "doc2", "doc3"])
            engine.database_manager.get_all_keys.assert_called_once()
    
    def test_validate_connection_success(self):
        """接続検証の正常ケースをテスト"""
        with patch.object(QueryEngine, '__init__', lambda x, **kwargs: None):
            engine = QueryEngine()
            
            # モックコンポーネントを設定
            engine.database_manager = Mock()
            engine.vector_embedder = Mock()
            
            engine.database_manager.get_vector_count.return_value = 5
            engine.vector_embedder.get_model_info.return_value = {
                "model_loaded": True,
                "model_name": "test-model"
            }
            
            # テスト実行
            result = engine.validate_connection()
            
            # 結果を検証
            self.assertTrue(result)
    
    def test_validate_connection_model_not_loaded(self):
        """モデル未読み込み時の接続検証エラーテスト"""
        with patch.object(QueryEngine, '__init__', lambda x, **kwargs: None):
            engine = QueryEngine()
            
            # モックコンポーネントを設定
            engine.database_manager = Mock()
            engine.vector_embedder = Mock()
            
            engine.database_manager.get_vector_count.return_value = 5
            engine.vector_embedder.get_model_info.return_value = {
                "model_loaded": False,
                "model_name": "test-model"
            }
            
            # RuntimeErrorが発生することを確認
            with self.assertRaises(RuntimeError) as context:
                engine.validate_connection()
            
            self.assertIn("QueryEngine接続検証に失敗", str(context.exception))


class TestQueryEngineIntegration(unittest.TestCase):
    """QueryEngineの統合テスト（実際のコンポーネントを使用）"""
    
    def setUp(self):
        """統合テスト用の初期化処理"""
        # テスト用の一時データベースファイル
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        # ログレベルを設定
        logging.getLogger().setLevel(logging.WARNING)
    
    def tearDown(self):
        """統合テスト用のクリーンアップ処理"""
        try:
            if os.path.exists(self.db_path):
                os.unlink(self.db_path)
        except PermissionError:
            # Windowsでファイルが使用中の場合はスキップ
            pass
    
    @patch('src.vector_embedder.SentenceTransformer')
    def test_integration_basic_flow(self, mock_sentence_transformer):
        """基本的な統合フローのテスト"""
        # SentenceTransformerのモックを設定
        mock_model = Mock()
        mock_model.encode.return_value = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_sentence_transformer.return_value = mock_model
        
        try:
            # QueryEngineを初期化
            engine = QueryEngine(db_path=self.db_path)
            
            # テストデータをデータベースに保存
            test_vectors = {
                "doc1": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                "doc2": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "doc3": np.array([0.5, 0.5, 0.0], dtype=np.float32)
            }
            
            for key, vector in test_vectors.items():
                engine.database_manager.store_vector(key, vector)
            
            # テキスト検索をテスト
            results = engine.search_by_text("テストクエリ", top_k=2)
            self.assertIsInstance(results, list)
            self.assertLessEqual(len(results), 2)
            
            # ドキュメント検索をテスト
            results = engine.search_by_document_key("doc1", top_k=2)
            self.assertIsInstance(results, list)
            self.assertLessEqual(len(results), 2)
            
            # 統計情報取得をテスト
            stats = engine.get_database_stats()
            self.assertEqual(stats["total_documents"], 3)
            
            # 接続検証をテスト
            self.assertTrue(engine.validate_connection())
            
        finally:
            # データベース接続を明示的に閉じる
            if 'engine' in locals():
                del engine


if __name__ == '__main__':
    unittest.main()