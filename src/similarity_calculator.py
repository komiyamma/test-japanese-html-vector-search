"""
類似度計算エンジン

このモジュールは、ベクトル間のコサイン類似度計算と
類似ドキュメント検索機能を提供します。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """
    ベクトル間の類似度計算と類似ドキュメント検索を行うクラス
    """
    
    def cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        2つのベクトル間のコサイン類似度を計算する
        
        Args:
            vector1: 第1のベクトル
            vector2: 第2のベクトル
            
        Returns:
            float: 0から1の間のコサイン類似度スコア（1は同一コンテンツを示す）
            
        Raises:
            ValueError: ベクトルが無効な場合
        """
        # 入力検証
        if vector1 is None or vector2 is None:
            raise ValueError("ベクトルがNoneです")
            
        if len(vector1) == 0 or len(vector2) == 0:
            raise ValueError("空のベクトルです")
            
        if vector1.shape != vector2.shape:
            raise ValueError(f"ベクトルの次元が一致しません: {vector1.shape} vs {vector2.shape}")
        
        # ゼロベクトルのチェック
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            logger.warning("ゼロベクトルが検出されました。類似度は0を返します。")
            return 0.0
        
        # コサイン類似度計算
        dot_product = np.dot(vector1, vector2)
        similarity = dot_product / (norm1 * norm2)
        
        # 数値精度の問題で1を超える場合があるため、クリップする
        similarity = np.clip(similarity, -1.0, 1.0)
        
        # -1から1の範囲を0から1の範囲に正規化
        normalized_similarity = (similarity + 1.0) / 2.0
        
        return float(normalized_similarity)
    
    def find_similar_documents(self, 
                             query_vector: np.ndarray, 
                             document_vectors: Dict[str, np.ndarray], 
                             top_k: int = 5,
                             threshold: float = 0.0) -> List[Tuple[str, float]]:
        """
        クエリベクトルに類似したドキュメントを検索する
        
        Args:
            query_vector: 検索クエリのベクトル
            document_vectors: ドキュメントキーとベクトルのマッピング
            top_k: 返す結果の最大数
            threshold: 類似度の最小閾値
            
        Returns:
            List[Tuple[str, float]]: (ドキュメントキー, 類似度スコア)のリスト
                                   類似度スコアでソート済み（降順）
                                   
        Raises:
            ValueError: 入力が無効な場合
        """
        if query_vector is None:
            raise ValueError("クエリベクトルがNoneです")
            
        if not document_vectors:
            logger.info("ドキュメントベクトルが空です")
            return []
            
        if top_k <= 0:
            raise ValueError("top_kは正の整数である必要があります")
            
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("thresholdは0.0から1.0の間である必要があります")
        
        similarities = []
        
        try:
            # 各ドキュメントベクトルとの類似度を計算
            for doc_key, doc_vector in document_vectors.items():
                try:
                    similarity = self.cosine_similarity(query_vector, doc_vector)
                    
                    # 閾値チェック
                    if similarity >= threshold:
                        similarities.append((doc_key, similarity))
                        
                except Exception as e:
                    logger.warning(f"ドキュメント '{doc_key}' の類似度計算でエラー: {e}")
                    continue
            
            # 類似度でソート（降順）
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # top_k個の結果を返す
            result = similarities[:top_k]
            
            logger.info(f"類似ドキュメント検索完了: {len(result)}件の結果")
            return result
            
        except Exception as e:
            logger.error(f"類似ドキュメント検索でエラー: {e}")
            raise
    
    def batch_similarity_calculation(self, 
                                   vectors: Dict[str, np.ndarray],
                                   batch_size: int = 100) -> Dict[Tuple[str, str], float]:
        """
        大量のベクトル間の類似度を効率的に計算する（パフォーマンス最適化）
        
        Args:
            vectors: ドキュメントキーとベクトルのマッピング
            batch_size: バッチ処理のサイズ
            
        Returns:
            Dict[Tuple[str, str], float]: (key1, key2) -> similarity のマッピング
        """
        if not vectors:
            return {}
            
        keys = list(vectors.keys())
        n = len(keys)
        similarities = {}
        
        logger.info(f"バッチ類似度計算開始: {n}個のドキュメント")
        
        # 効率的な計算のため、ベクトルを行列に変換
        vector_matrix = np.array([vectors[key] for key in keys])
        
        try:
            # 正規化
            norms = np.linalg.norm(vector_matrix, axis=1, keepdims=True)
            # ゼロベクトルの処理 - 無限大や無効な値をチェック
            if np.any(np.isinf(vector_matrix)) or np.any(np.isnan(vector_matrix)):
                raise ValueError("ベクトルに無効な値（無限大またはNaN）が含まれています")
            
            # ゼロベクトルの処理
            norms = np.where(norms == 0, 1, norms)
            normalized_vectors = vector_matrix / norms
            
            # 類似度行列を計算
            similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
            
            # 結果を辞書に変換
            for i in range(n):
                for j in range(i + 1, n):  # 上三角行列のみ計算（対称性を利用）
                    key1, key2 = keys[i], keys[j]
                    similarity = float((similarity_matrix[i, j] + 1.0) / 2.0)  # 正規化
                    similarities[(key1, key2)] = similarity
                    similarities[(key2, key1)] = similarity  # 対称性
            
            logger.info(f"バッチ類似度計算完了: {len(similarities)}個のペア")
            return similarities
            
        except Exception as e:
            logger.error(f"バッチ類似度計算でエラー: {e}")
            raise