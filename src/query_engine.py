"""
QueryEngine - ベクトルデータベースのクエリエンジン

このモジュールは、テキストクエリやドキュメントキーによる
類似ドキュメント検索機能を提供します。
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Dict
from .vector_embedder import VectorEmbedder
from .database_manager import DatabaseManager
from .similarity_calculator import SimilarityCalculator

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    ベクトルデータベースに対するクエリ処理を行うクラス
    
    テキストクエリのベクトル化、ドキュメントキーによる類似検索、
    結果数制限、閾値フィルタリング機能を提供します。
    """
    
    def __init__(self, 
                 db_path: str = "vectors.db",
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        QueryEngineを初期化
        
        Args:
            db_path (str): データベースファイルのパス
            model_name (str): 使用するSentence Transformersモデル名
            
        Raises:
            RuntimeError: 初期化に失敗した場合
        """
        self.db_path = db_path
        self.model_name = model_name
        
        try:
            # コンポーネントを初期化
            self.vector_embedder = VectorEmbedder(model_name)
            self.database_manager = DatabaseManager(db_path)
            self.similarity_calculator = SimilarityCalculator()
            
            logger.info(f"QueryEngine初期化完了 (DB: {db_path}, Model: {model_name})")
            
        except Exception as e:
            error_msg = f"QueryEngine初期化に失敗: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def search_by_text(self, 
                      query_text: str, 
                      top_k: int = 5, 
                      threshold: float = 0.0) -> List[Tuple[str, float]]:
        """
        テキストクエリによる類似ドキュメント検索
        
        Args:
            query_text (str): 検索クエリテキスト
            top_k (int): 返す結果の最大数（デフォルト: 5）
            threshold (float): 類似度の最小閾値（デフォルト: 0.0）
            
        Returns:
            List[Tuple[str, float]]: (ドキュメントキー, 類似度スコア)のリスト
                                   類似度スコアでソート済み（降順）
                                   
        Raises:
            ValueError: 入力パラメータが無効な場合
            RuntimeError: 検索処理に失敗した場合
        """
        # 入力検証
        if not query_text or not query_text.strip():
            raise ValueError("検索クエリテキストが空です")
        
        if top_k <= 0:
            raise ValueError("top_kは正の整数である必要があります")
        
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("thresholdは0.0から1.0の間である必要があります")
        
        try:
            logger.info(f"テキスト検索開始: '{query_text[:50]}...' (top_k={top_k}, threshold={threshold})")
            
            # クエリテキストをベクトル化
            query_vector = self.vector_embedder.embed_text(query_text)
            logger.debug(f"クエリベクトル化完了 (次元: {query_vector.shape[0]})")
            
            # データベースから全ベクトルを取得
            try:
                document_vectors = self.database_manager.get_all_vectors()
                if not document_vectors:
                    logger.warning("データベースにベクトルデータが存在しません")
                    return []
                
                logger.debug(f"データベースから {len(document_vectors)} 個のベクトルを取得")
                
            except Exception as e:
                error_msg = f"データベースからのベクトル取得に失敗: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
            
            # 類似ドキュメントを検索
            similar_docs = self.similarity_calculator.find_similar_documents(
                query_vector=query_vector,
                document_vectors=document_vectors,
                top_k=top_k,
                threshold=threshold
            )
            
            logger.info(f"テキスト検索完了: {len(similar_docs)} 件の結果")
            return similar_docs
            
        except ValueError:
            # 入力検証エラーは再発生
            raise
        except Exception as e:
            error_msg = f"テキスト検索処理に失敗: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def search_by_document_key(self, 
                              document_key: str, 
                              top_k: int = 5, 
                              threshold: float = 0.0,
                              exclude_self: bool = True) -> List[Tuple[str, float]]:
        """
        ドキュメントキーによる類似ドキュメント検索
        
        Args:
            document_key (str): 基準となるドキュメントキー
            top_k (int): 返す結果の最大数（デフォルト: 5）
            threshold (float): 類似度の最小閾値（デフォルト: 0.0）
            exclude_self (bool): 自分自身を結果から除外するか（デフォルト: True）
            
        Returns:
            List[Tuple[str, float]]: (ドキュメントキー, 類似度スコア)のリスト
                                   類似度スコアでソート済み（降順）
                                   
        Raises:
            ValueError: 入力パラメータが無効な場合
            RuntimeError: 検索処理に失敗した場合
        """
        # 入力検証
        if not document_key or not document_key.strip():
            raise ValueError("ドキュメントキーが空です")
        
        if top_k <= 0:
            raise ValueError("top_kは正の整数である必要があります")
        
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("thresholdは0.0から1.0の間である必要があります")
        
        try:
            logger.info(f"ドキュメント検索開始: '{document_key}' (top_k={top_k}, threshold={threshold})")
            
            # 基準ドキュメントのベクトルを取得
            try:
                query_vector = self.database_manager.get_vector(document_key)
                if query_vector is None:
                    raise ValueError(f"指定されたドキュメントキーが見つかりません: {document_key}")
                
                logger.debug(f"基準ベクトル取得完了 (次元: {query_vector.shape[0]})")
                
            except Exception as e:
                if "見つかりません" in str(e):
                    raise ValueError(str(e))
                error_msg = f"基準ドキュメントのベクトル取得に失敗: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
            
            # データベースから全ベクトルを取得
            try:
                document_vectors = self.database_manager.get_all_vectors()
                if not document_vectors:
                    logger.warning("データベースにベクトルデータが存在しません")
                    return []
                
                logger.debug(f"データベースから {len(document_vectors)} 個のベクトルを取得")
                
            except Exception as e:
                error_msg = f"データベースからのベクトル取得に失敗: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
            
            # 自分自身を除外する場合
            if exclude_self and document_key in document_vectors:
                document_vectors = {k: v for k, v in document_vectors.items() if k != document_key}
                logger.debug(f"自分自身を除外: {len(document_vectors)} 個のベクトルで検索")
            
            # 類似ドキュメントを検索
            similar_docs = self.similarity_calculator.find_similar_documents(
                query_vector=query_vector,
                document_vectors=document_vectors,
                top_k=top_k,
                threshold=threshold
            )
            
            logger.info(f"ドキュメント検索完了: {len(similar_docs)} 件の結果")
            return similar_docs
            
        except ValueError:
            # 入力検証エラーは再発生
            raise
        except Exception as e:
            error_msg = f"ドキュメント検索処理に失敗: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def get_document_info(self, document_key: str) -> Optional[Dict]:
        """
        指定されたドキュメントの情報を取得
        
        Args:
            document_key (str): ドキュメントキー
            
        Returns:
            Optional[Dict]: ドキュメント情報（存在しない場合はNone）
                          - key: ドキュメントキー
                          - vector_dimension: ベクトル次元数
                          - exists: 存在フラグ
                          
        Raises:
            RuntimeError: データベース接続エラー
        """
        if not document_key or not document_key.strip():
            return None
        
        try:
            vector = self.database_manager.get_vector(document_key)
            if vector is None:
                return {
                    "key": document_key,
                    "exists": False,
                    "vector_dimension": None
                }
            
            return {
                "key": document_key,
                "exists": True,
                "vector_dimension": vector.shape[0]
            }
            
        except Exception as e:
            error_msg = f"ドキュメント情報取得に失敗: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def get_database_stats(self) -> Dict:
        """
        データベースの統計情報を取得
        
        Returns:
            Dict: データベース統計情報
                - total_documents: 総ドキュメント数
                - database_path: データベースファイルパス
                - model_name: 使用モデル名
                - vector_dimension: ベクトル次元数
                
        Raises:
            RuntimeError: データベース接続エラー
        """
        try:
            total_docs = self.database_manager.get_vector_count()
            vector_dim = self.vector_embedder.get_embedding_dimension()
            
            return {
                "total_documents": total_docs,
                "database_path": self.db_path,
                "model_name": self.model_name,
                "vector_dimension": vector_dim
            }
            
        except Exception as e:
            error_msg = f"データベース統計情報取得に失敗: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def list_all_documents(self) -> List[str]:
        """
        データベース内のすべてのドキュメントキーを取得
        
        Returns:
            List[str]: ドキュメントキーのリスト（ソート済み）
            
        Raises:
            RuntimeError: データベース接続エラー
        """
        try:
            keys = self.database_manager.get_all_keys()
            logger.debug(f"ドキュメントキー一覧取得: {len(keys)} 件")
            return keys
            
        except Exception as e:
            error_msg = f"ドキュメントキー一覧取得に失敗: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def validate_connection(self) -> bool:
        """
        データベース接続とモデルの状態を検証
        
        Returns:
            bool: すべてのコンポーネントが正常な場合True
            
        Raises:
            RuntimeError: 検証中にエラーが発生した場合
        """
        try:
            # データベース接続テスト
            count = self.database_manager.get_vector_count()
            logger.debug(f"データベース接続OK: {count} 件のドキュメント")
            
            # モデル状態テスト
            model_info = self.vector_embedder.get_model_info()
            if not model_info["model_loaded"]:
                raise RuntimeError("ベクトル化モデルが読み込まれていません")
            
            logger.debug(f"モデル状態OK: {model_info['model_name']}")
            
            logger.info("QueryEngine接続検証完了")
            return True
            
        except Exception as e:
            error_msg = f"QueryEngine接続検証に失敗: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e