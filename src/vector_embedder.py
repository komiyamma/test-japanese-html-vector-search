"""
日本語テキストのベクトル化を行うVectorEmbedderクラス

このモジュールは、Sentence Transformersライブラリを使用して
日本語テキストをベクトル埋め込みに変換する機能を提供します。
"""

import logging
import numpy as np
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer
import torch

# ログ設定
logger = logging.getLogger(__name__)


class VectorEmbedder:
    """
    日本語テキストのベクトル化を行うクラス
    
    Sentence Transformersライブラリを使用して、日本語テキストを
    高次元ベクトル空間にマッピングします。メモリ効率を考慮した
    バッチ処理機能も提供します。
    """
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        VectorEmbedderを初期化
        
        Args:
            model_name (str): 使用するSentence Transformersモデル名
                            デフォルトは多言語対応のMiniLMモデル
        
        Raises:
            RuntimeError: モデルの読み込みに失敗した場合
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            logger.info(f"Sentence Transformersモデルを読み込み中: {model_name}")
            logger.info(f"使用デバイス: {self.device}")
            
            self.model = SentenceTransformer(model_name, device=self.device)
            
            # モデル情報をログ出力
            logger.info(f"モデル読み込み完了")
            logger.info(f"ベクトル次元数: {self.model.get_sentence_embedding_dimension()}")
            
        except Exception as e:
            error_msg = f"Sentence Transformersモデルの読み込みに失敗: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        単一のテキストをベクトル化
        
        Args:
            text (str): ベクトル化するテキスト
            
        Returns:
            np.ndarray: テキストのベクトル表現（float32形式）
            
        Raises:
            ValueError: 入力テキストが空の場合
            RuntimeError: ベクトル化処理に失敗した場合
        """
        if not text or not text.strip():
            raise ValueError("入力テキストが空です")
        
        if self.model is None:
            raise RuntimeError("モデルが初期化されていません")
        
        try:
            logger.debug(f"テキストをベクトル化中（長さ: {len(text)}文字）")
            
            # テキストをベクトル化
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,  # 正規化してコサイン類似度計算を効率化
                show_progress_bar=False
            )
            
            # float32に変換してメモリ効率を向上
            embedding = embedding.astype(np.float32)
            
            logger.debug(f"ベクトル化完了（次元数: {embedding.shape[0]}）")
            return embedding
            
        except Exception as e:
            error_msg = f"テキストのベクトル化に失敗: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        複数のテキストを効率的にバッチベクトル化
        
        Args:
            texts (List[str]): ベクトル化するテキストのリスト
            batch_size (int): バッチサイズ（デフォルト: 32）
                            メモリ使用量に応じて調整可能
            
        Returns:
            List[np.ndarray]: 各テキストのベクトル表現のリスト
            
        Raises:
            ValueError: 入力リストが空の場合
            RuntimeError: バッチベクトル化処理に失敗した場合
        """
        if not texts:
            raise ValueError("入力テキストリストが空です")
        
        if self.model is None:
            raise RuntimeError("モデルが初期化されていません")
        
        # 空のテキストを除外し、インデックスを記録
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
            else:
                logger.warning(f"インデックス {i} のテキストが空のためスキップします")
        
        if not valid_texts:
            raise ValueError("有効なテキストが見つかりません")
        
        try:
            logger.info(f"バッチベクトル化開始: {len(valid_texts)}件のテキスト")
            logger.info(f"バッチサイズ: {batch_size}")
            
            # メモリ効率を考慮したバッチ処理
            all_embeddings = []
            
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i + batch_size]
                logger.debug(f"バッチ {i//batch_size + 1}/{(len(valid_texts) + batch_size - 1)//batch_size} を処理中")
                
                try:
                    # バッチベクトル化実行
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        batch_size=min(batch_size, len(batch_texts))
                    )
                    
                    # float32に変換
                    batch_embeddings = batch_embeddings.astype(np.float32)
                    
                    # リストに追加
                    for embedding in batch_embeddings:
                        all_embeddings.append(embedding)
                        
                except Exception as e:
                    logger.error(f"バッチ {i//batch_size + 1} の処理中にエラー: {e}")
                    # メモリ不足の場合はバッチサイズを半分にして再試行
                    if "out of memory" in str(e).lower() and batch_size > 1:
                        logger.warning(f"メモリ不足のためバッチサイズを {batch_size//2} に縮小して再試行")
                        return self.embed_batch(texts, batch_size//2)
                    else:
                        raise
            
            # 元の順序に合わせて結果を構築（空テキストの位置にはゼロベクトルを挿入）
            result_embeddings = []
            embedding_dim = all_embeddings[0].shape[0] if all_embeddings else 384
            valid_idx = 0
            
            for i in range(len(texts)):
                if i in valid_indices:
                    result_embeddings.append(all_embeddings[valid_idx])
                    valid_idx += 1
                else:
                    # 空テキストの場合はゼロベクトルを挿入
                    zero_vector = np.zeros(embedding_dim, dtype=np.float32)
                    result_embeddings.append(zero_vector)
                    logger.warning(f"インデックス {i} にゼロベクトルを挿入")
            
            logger.info(f"バッチベクトル化完了: {len(result_embeddings)}件")
            return result_embeddings
            
        except Exception as e:
            error_msg = f"バッチベクトル化に失敗: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def get_embedding_dimension(self) -> int:
        """
        ベクトルの次元数を取得
        
        Returns:
            int: ベクトルの次元数
            
        Raises:
            RuntimeError: モデルが初期化されていない場合
        """
        if self.model is None:
            raise RuntimeError("モデルが初期化されていません")
        
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_info(self) -> dict:
        """
        モデル情報を取得
        
        Returns:
            dict: モデル名、次元数、デバイス情報を含む辞書
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.get_embedding_dimension() if self.model else None,
            "device": self.device,
            "model_loaded": self.model is not None
        }