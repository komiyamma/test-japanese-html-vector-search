"""
VectorEmbedderクラスの単体テスト

日本語テキストのベクトル化機能をテストします。
"""

import unittest
import numpy as np
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# テスト対象のモジュールをインポート
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.vector_embedder import VectorEmbedder


class TestVectorEmbedder(unittest.TestCase):
    """VectorEmbedderクラスのテストケース"""
    
    def setUp(self):
        """各テストの前に実行される初期化処理"""
        # テスト用の軽量モデルを使用（実際のダウンロードを避けるためモックを使用）
        self.test_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        
    @patch('src.vector_embedder.SentenceTransformer')
    @patch('src.vector_embedder.torch')
    def test_init_success(self, mock_torch, mock_sentence_transformer):
        """正常な初期化のテスト"""
        # モックの設定
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        # VectorEmbedderを初期化
        embedder = VectorEmbedder(self.test_model_name)
        
        # 検証
        self.assertEqual(embedder.model_name, self.test_model_name)
        self.assertEqual(embedder.device, "cpu")
        self.assertIsNotNone(embedder.model)
        mock_sentence_transformer.assert_called_once_with(self.test_model_name, device="cpu")
    
    @patch('src.vector_embedder.SentenceTransformer')
    @patch('src.vector_embedder.torch')
    def test_init_with_cuda(self, mock_torch, mock_sentence_transformer):
        """CUDA利用可能時の初期化テスト"""
        # モックの設定
        mock_torch.cuda.is_available.return_value = True
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        # VectorEmbedderを初期化
        embedder = VectorEmbedder(self.test_model_name)
        
        # 検証
        self.assertEqual(embedder.device, "cuda")
        mock_sentence_transformer.assert_called_once_with(self.test_model_name, device="cuda")
    
    @patch('src.vector_embedder.SentenceTransformer')
    @patch('src.vector_embedder.torch')
    def test_init_failure(self, mock_torch, mock_sentence_transformer):
        """モデル読み込み失敗時のテスト"""
        # モックの設定
        mock_torch.cuda.is_available.return_value = False
        mock_sentence_transformer.side_effect = Exception("モデル読み込みエラー")
        
        # 例外が発生することを確認
        with self.assertRaises(RuntimeError) as context:
            VectorEmbedder(self.test_model_name)
        
        self.assertIn("Sentence Transformersモデルの読み込みに失敗", str(context.exception))
    
    @patch('src.vector_embedder.SentenceTransformer')
    @patch('src.vector_embedder.torch')
    def test_embed_text_success(self, mock_torch, mock_sentence_transformer):
        """単一テキストのベクトル化成功テスト"""
        # モックの設定
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        
        # テスト用のベクトルを作成
        test_vector = np.random.rand(384).astype(np.float64)
        mock_model.encode.return_value = test_vector
        mock_sentence_transformer.return_value = mock_model
        
        # VectorEmbedderを初期化
        embedder = VectorEmbedder(self.test_model_name)
        
        # テスト実行
        test_text = "これは日本語のテストテキストです。"
        result = embedder.embed_text(test_text)
        
        # 検証
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, (384,))
        
        # モデルのencodeメソッドが正しい引数で呼ばれたことを確認
        mock_model.encode.assert_called_once_with(
            test_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
    
    @patch('src.vector_embedder.SentenceTransformer')
    @patch('src.vector_embedder.torch')
    def test_embed_text_empty_input(self, mock_torch, mock_sentence_transformer):
        """空のテキスト入力時のエラーテスト"""
        # モックの設定
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        # VectorEmbedderを初期化
        embedder = VectorEmbedder(self.test_model_name)
        
        # 空文字列でのテスト
        with self.assertRaises(ValueError) as context:
            embedder.embed_text("")
        self.assertIn("入力テキストが空です", str(context.exception))
        
        # 空白のみの文字列でのテスト
        with self.assertRaises(ValueError) as context:
            embedder.embed_text("   ")
        self.assertIn("入力テキストが空です", str(context.exception))
    
    @patch('src.vector_embedder.SentenceTransformer')
    @patch('src.vector_embedder.torch')
    def test_embed_text_model_error(self, mock_torch, mock_sentence_transformer):
        """ベクトル化処理エラーのテスト"""
        # モックの設定
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.side_effect = Exception("エンコードエラー")
        mock_sentence_transformer.return_value = mock_model
        
        # VectorEmbedderを初期化
        embedder = VectorEmbedder(self.test_model_name)
        
        # エラーが発生することを確認
        with self.assertRaises(RuntimeError) as context:
            embedder.embed_text("テストテキスト")
        
        self.assertIn("テキストのベクトル化に失敗", str(context.exception))
    
    @patch('src.vector_embedder.SentenceTransformer')
    @patch('src.vector_embedder.torch')
    def test_embed_batch_success(self, mock_torch, mock_sentence_transformer):
        """バッチベクトル化成功テスト"""
        # モックの設定
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        
        # テスト用のベクトル配列を作成
        test_vectors = np.random.rand(3, 384).astype(np.float64)
        mock_model.encode.return_value = test_vectors
        mock_sentence_transformer.return_value = mock_model
        
        # VectorEmbedderを初期化
        embedder = VectorEmbedder(self.test_model_name)
        
        # テスト実行
        test_texts = [
            "最初の日本語テキストです。",
            "二番目のテキストです。",
            "三番目のテキストです。"
        ]
        results = embedder.embed_batch(test_texts)
        
        # 検証
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.dtype, np.float32)
            self.assertEqual(result.shape, (384,))
    
    @patch('src.vector_embedder.SentenceTransformer')
    @patch('src.vector_embedder.torch')
    def test_embed_batch_with_empty_texts(self, mock_torch, mock_sentence_transformer):
        """空テキストを含むバッチベクトル化テスト"""
        # モックの設定
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        
        # 有効なテキストのみのベクトル配列を作成
        test_vectors = np.random.rand(2, 384).astype(np.float64)
        mock_model.encode.return_value = test_vectors
        mock_sentence_transformer.return_value = mock_model
        
        # VectorEmbedderを初期化
        embedder = VectorEmbedder(self.test_model_name)
        
        # テスト実行（空テキストを含む）
        test_texts = [
            "有効なテキスト1",
            "",  # 空テキスト
            "有効なテキスト2",
            "   "  # 空白のみ
        ]
        results = embedder.embed_batch(test_texts)
        
        # 検証
        self.assertEqual(len(results), 4)
        
        # 有効なテキストの結果を確認
        self.assertEqual(results[0].shape, (384,))
        self.assertEqual(results[2].shape, (384,))
        
        # 空テキストの結果（ゼロベクトル）を確認
        self.assertTrue(np.allclose(results[1], np.zeros(384)))
        self.assertTrue(np.allclose(results[3], np.zeros(384)))
    
    @patch('src.vector_embedder.SentenceTransformer')
    @patch('src.vector_embedder.torch')
    def test_embed_batch_empty_list(self, mock_torch, mock_sentence_transformer):
        """空リストでのバッチベクトル化エラーテスト"""
        # モックの設定
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        # VectorEmbedderを初期化
        embedder = VectorEmbedder(self.test_model_name)
        
        # 空リストでのテスト
        with self.assertRaises(ValueError) as context:
            embedder.embed_batch([])
        self.assertIn("入力テキストリストが空です", str(context.exception))
    
    @patch('src.vector_embedder.SentenceTransformer')
    @patch('src.vector_embedder.torch')
    def test_embed_batch_memory_error_retry(self, mock_torch, mock_sentence_transformer):
        """メモリ不足時の自動リトライテスト"""
        # モックの設定
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        
        # 最初の呼び出しでメモリエラー、2回目で成功
        test_vectors = np.random.rand(2, 384).astype(np.float64)
        mock_model.encode.side_effect = [
            Exception("CUDA out of memory"),  # 最初はメモリエラー
            test_vectors  # 2回目は成功
        ]
        mock_sentence_transformer.return_value = mock_model
        
        # VectorEmbedderを初期化
        embedder = VectorEmbedder(self.test_model_name)
        
        # テスト実行
        test_texts = ["テキスト1", "テキスト2"]
        results = embedder.embed_batch(test_texts, batch_size=4)
        
        # 検証（リトライが成功したことを確認）
        self.assertEqual(len(results), 2)
        self.assertEqual(mock_model.encode.call_count, 2)  # 2回呼ばれたことを確認
    
    @patch('src.vector_embedder.SentenceTransformer')
    @patch('src.vector_embedder.torch')
    def test_get_embedding_dimension(self, mock_torch, mock_sentence_transformer):
        """ベクトル次元数取得テスト"""
        # モックの設定
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        # VectorEmbedderを初期化
        embedder = VectorEmbedder(self.test_model_name)
        
        # テスト実行
        dimension = embedder.get_embedding_dimension()
        
        # 検証
        self.assertEqual(dimension, 384)
    
    @patch('src.vector_embedder.SentenceTransformer')
    @patch('src.vector_embedder.torch')
    def test_get_model_info(self, mock_torch, mock_sentence_transformer):
        """モデル情報取得テスト"""
        # モックの設定
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        # VectorEmbedderを初期化
        embedder = VectorEmbedder(self.test_model_name)
        
        # テスト実行
        info = embedder.get_model_info()
        
        # 検証
        expected_info = {
            "model_name": self.test_model_name,
            "embedding_dimension": 384,
            "device": "cpu",
            "model_loaded": True
        }
        self.assertEqual(info, expected_info)


class TestVectorEmbedderIntegration(unittest.TestCase):
    """VectorEmbedderの統合テスト（実際のモデルを使用）"""
    
    @unittest.skipIf(os.environ.get('SKIP_INTEGRATION_TESTS'), "統合テストをスキップ")
    def test_real_model_japanese_text(self):
        """実際のモデルを使用した日本語テキストのテスト"""
        try:
            # 軽量なモデルを使用
            embedder = VectorEmbedder("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            
            # 日本語テキストのテスト
            japanese_texts = [
                "これは日本語のテストです。",
                "今日は良い天気ですね。",
                "プログラミングは楽しいです。"
            ]
            
            # 単一テキストのベクトル化
            single_result = embedder.embed_text(japanese_texts[0])
            self.assertIsInstance(single_result, np.ndarray)
            self.assertEqual(single_result.dtype, np.float32)
            
            # バッチベクトル化
            batch_results = embedder.embed_batch(japanese_texts)
            self.assertEqual(len(batch_results), 3)
            
            # 各結果が適切な形状を持つことを確認
            for result in batch_results:
                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(result.dtype, np.float32)
                self.assertEqual(result.shape, single_result.shape)
            
        except Exception as e:
            self.skipTest(f"統合テストをスキップ（モデルダウンロードエラー）: {e}")


if __name__ == '__main__':
    # ログレベルを設定
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # テスト実行
    unittest.main()