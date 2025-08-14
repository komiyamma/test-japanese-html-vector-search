"""
SimilarityCalculator クラスの単体テスト
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# テスト対象のモジュールをインポートするためのパス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from similarity_calculator import SimilarityCalculator


class TestSimilarityCalculator(unittest.TestCase):
    """SimilarityCalculator クラスのテストケース"""
    
    def setUp(self):
        """テストの前準備"""
        self.calculator = SimilarityCalculator()
        
        # テスト用のベクトル
        self.vector1 = np.array([1.0, 0.0, 0.0])
        self.vector2 = np.array([0.0, 1.0, 0.0])
        self.vector3 = np.array([1.0, 0.0, 0.0])  # vector1と同じ
        self.zero_vector = np.array([0.0, 0.0, 0.0])
        
        # より複雑なテスト用ベクトル
        self.complex_vector1 = np.array([0.5, 0.5, 0.7071])
        self.complex_vector2 = np.array([0.7071, 0.7071, 0.0])
    
    def test_cosine_similarity_identical_vectors(self):
        """同一ベクトルの類似度は1.0になることをテスト"""
        similarity = self.calculator.cosine_similarity(self.vector1, self.vector3)
        self.assertAlmostEqual(similarity, 1.0, places=5)
    
    def test_cosine_similarity_orthogonal_vectors(self):
        """直交ベクトルの類似度は0.5になることをテスト（正規化後）"""
        similarity = self.calculator.cosine_similarity(self.vector1, self.vector2)
        self.assertAlmostEqual(similarity, 0.5, places=5)
    
    def test_cosine_similarity_zero_vector(self):
        """ゼロベクトルとの類似度は0.0になることをテスト"""
        with patch('similarity_calculator.logger') as mock_logger:
            similarity = self.calculator.cosine_similarity(self.vector1, self.zero_vector)
            self.assertEqual(similarity, 0.0)
            mock_logger.warning.assert_called_once()
    
    def test_cosine_similarity_none_input(self):
        """Noneが入力された場合のエラーハンドリングをテスト"""
        with self.assertRaises(ValueError) as context:
            self.calculator.cosine_similarity(None, self.vector1)
        self.assertIn("ベクトルがNoneです", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            self.calculator.cosine_similarity(self.vector1, None)
        self.assertIn("ベクトルがNoneです", str(context.exception))
    
    def test_cosine_similarity_empty_vector(self):
        """空のベクトルが入力された場合のエラーハンドリングをテスト"""
        empty_vector = np.array([])
        with self.assertRaises(ValueError) as context:
            self.calculator.cosine_similarity(empty_vector, self.vector1)
        self.assertIn("空のベクトルです", str(context.exception))
    
    def test_cosine_similarity_dimension_mismatch(self):
        """次元が異なるベクトルのエラーハンドリングをテスト"""
        vector_2d = np.array([1.0, 0.0])
        with self.assertRaises(ValueError) as context:
            self.calculator.cosine_similarity(self.vector1, vector_2d)
        self.assertIn("ベクトルの次元が一致しません", str(context.exception))
    
    def test_cosine_similarity_complex_vectors(self):
        """複雑なベクトルでの類似度計算をテスト"""
        similarity = self.calculator.cosine_similarity(self.complex_vector1, self.complex_vector2)
        # 手動計算: dot_product = 0.5*0.7071 + 0.5*0.7071 + 0.7071*0 = 0.7071
        # norm1 = sqrt(0.25 + 0.25 + 0.5) = 1.0
        # norm2 = sqrt(0.5 + 0.5 + 0) = 1.0
        # cosine = 0.7071, normalized = (0.7071 + 1) / 2 = 0.8536
        expected = (0.7071 + 1.0) / 2.0
        self.assertAlmostEqual(similarity, expected, places=3)
    
    def test_find_similar_documents_basic(self):
        """基本的な類似ドキュメント検索をテスト"""
        document_vectors = {
            "doc1": self.vector1,
            "doc2": self.vector2,
            "doc3": self.vector3,
        }
        
        results = self.calculator.find_similar_documents(
            self.vector1, document_vectors, top_k=3
        )
        
        # 結果が3つ返されることを確認
        self.assertEqual(len(results), 3)
        
        # doc1とdoc3は同一ベクトルなので、どちらも類似度1.0になる
        # 結果の順序は辞書の順序に依存する可能性があるため、類似度で確認
        high_similarity_docs = [doc for doc, sim in results if sim > 0.9]
        self.assertEqual(len(high_similarity_docs), 2)  # doc1とdoc3
        
        # 最高類似度は1.0であることを確認
        max_similarity = max(sim for _, sim in results)
        self.assertAlmostEqual(max_similarity, 1.0, places=5)
    
    def test_find_similar_documents_with_threshold(self):
        """閾値を使った類似ドキュメント検索をテスト"""
        document_vectors = {
            "doc1": self.vector1,
            "doc2": self.vector2,
            "doc3": self.vector3,
        }
        
        results = self.calculator.find_similar_documents(
            self.vector1, document_vectors, top_k=3, threshold=0.8
        )
        
        # 閾値0.8以上の結果のみ返される（doc3のみ）
        self.assertEqual(len(results), 2)  # doc1とdoc3
        for _, similarity in results:
            self.assertGreaterEqual(similarity, 0.8)
    
    def test_find_similar_documents_empty_input(self):
        """空の入力での類似ドキュメント検索をテスト"""
        results = self.calculator.find_similar_documents(
            self.vector1, {}, top_k=5
        )
        self.assertEqual(results, [])
    
    def test_find_similar_documents_none_query(self):
        """Noneクエリでのエラーハンドリングをテスト"""
        document_vectors = {"doc1": self.vector1}
        
        with self.assertRaises(ValueError) as context:
            self.calculator.find_similar_documents(None, document_vectors)
        self.assertIn("クエリベクトルがNoneです", str(context.exception))
    
    def test_find_similar_documents_invalid_top_k(self):
        """無効なtop_kでのエラーハンドリングをテスト"""
        document_vectors = {"doc1": self.vector1}
        
        with self.assertRaises(ValueError) as context:
            self.calculator.find_similar_documents(self.vector1, document_vectors, top_k=0)
        self.assertIn("top_kは正の整数である必要があります", str(context.exception))
    
    def test_find_similar_documents_invalid_threshold(self):
        """無効な閾値でのエラーハンドリングをテスト"""
        document_vectors = {"doc1": self.vector1}
        
        with self.assertRaises(ValueError) as context:
            self.calculator.find_similar_documents(
                self.vector1, document_vectors, threshold=-0.1
            )
        self.assertIn("thresholdは0.0から1.0の間である必要があります", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            self.calculator.find_similar_documents(
                self.vector1, document_vectors, threshold=1.1
            )
        self.assertIn("thresholdは0.0から1.0の間である必要があります", str(context.exception))
    
    def test_find_similar_documents_sorting(self):
        """結果が類似度でソートされることをテスト"""
        # 異なる類似度を持つベクトルを作成
        doc_vectors = {
            "low_sim": np.array([0.0, 1.0, 0.0]),      # 低い類似度
            "high_sim": np.array([1.0, 0.0, 0.0]),     # 高い類似度（同一）
            "med_sim": np.array([0.7071, 0.7071, 0.0]) # 中程度の類似度
        }
        
        results = self.calculator.find_similar_documents(
            np.array([1.0, 0.0, 0.0]), doc_vectors, top_k=3
        )
        
        # 類似度が降順でソートされていることを確認
        self.assertEqual(len(results), 3)
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i][1], results[i + 1][1])
    
    @patch('similarity_calculator.logger')
    def test_find_similar_documents_error_handling(self, mock_logger):
        """個別ドキュメントでのエラーハンドリングをテスト"""
        # 無効なベクトルを含むドキュメント集合
        document_vectors = {
            "valid_doc": self.vector1,
            "invalid_doc": np.array([])  # 空のベクトル
        }
        
        results = self.calculator.find_similar_documents(
            self.vector1, document_vectors, top_k=5
        )
        
        # 有効なドキュメントのみ結果に含まれる
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "valid_doc")
        
        # 警告がログに記録される
        mock_logger.warning.assert_called()
    
    def test_batch_similarity_calculation_basic(self):
        """基本的なバッチ類似度計算をテスト"""
        vectors = {
            "doc1": self.vector1,
            "doc2": self.vector2,
            "doc3": self.vector3
        }
        
        similarities = self.calculator.batch_similarity_calculation(vectors)
        
        # 期待される組み合わせ数: n*(n-1) = 3*2 = 6
        self.assertEqual(len(similarities), 6)
        
        # 対称性をテスト
        self.assertEqual(similarities[("doc1", "doc2")], similarities[("doc2", "doc1")])
        
        # 同一ドキュメント間の類似度をテスト
        self.assertAlmostEqual(similarities[("doc1", "doc3")], 1.0, places=3)
    
    def test_batch_similarity_calculation_empty_input(self):
        """空の入力でのバッチ類似度計算をテスト"""
        similarities = self.calculator.batch_similarity_calculation({})
        self.assertEqual(similarities, {})
    
    @patch('similarity_calculator.logger')
    def test_batch_similarity_calculation_error_handling(self, mock_logger):
        """バッチ類似度計算でのエラーハンドリングをテスト"""
        # 無効なベクトルを含む場合
        vectors = {
            "doc1": np.array([float('inf'), 0.0, 0.0]),  # 無限大を含む
            "doc2": self.vector1
        }
        
        with self.assertRaises(Exception):
            self.calculator.batch_similarity_calculation(vectors)
        
        mock_logger.error.assert_called()
    
    def test_performance_optimization(self):
        """パフォーマンス最適化のテスト（大量データ）"""
        # 大量のベクトルを生成
        n_docs = 100
        vectors = {}
        for i in range(n_docs):
            # ランダムなベクトルを生成
            np.random.seed(i)  # 再現性のため
            vector = np.random.rand(384).astype(np.float32)  # 実際のモデル次元
            vectors[f"doc_{i}"] = vector
        
        # バッチ処理が正常に完了することを確認
        similarities = self.calculator.batch_similarity_calculation(vectors, batch_size=50)
        
        # 結果の妥当性をチェック
        expected_pairs = n_docs * (n_docs - 1)
        self.assertEqual(len(similarities), expected_pairs)
        
        # すべての類似度が有効な範囲内にあることを確認
        for similarity in similarities.values():
            self.assertGreaterEqual(similarity, 0.0)
            self.assertLessEqual(similarity, 1.0)


if __name__ == '__main__':
    unittest.main()