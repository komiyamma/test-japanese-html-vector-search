"""
DatabaseManager - SQLiteデータベースでのベクトルデータ管理

このモジュールは、ベクトル化されたドキュメントデータをSQLiteデータベースに
保存・取得・更新する機能を提供します。
"""

import sqlite3
import pickle
import numpy as np
from typing import Optional, Dict, List, Tuple
import logging
from pathlib import Path


class DatabaseManager:
    """SQLiteデータベースでベクトルデータを管理するクラス"""
    
    def __init__(self, db_path: str = "vectors.db"):
        """
        DatabaseManagerを初期化
        
        Args:
            db_path (str): データベースファイルのパス
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._ensure_database_directory()
        self.create_table()
    
    def __enter__(self):
        """コンテキストマネージャーのエントリー"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了処理"""
        # 特別なクリーンアップは不要（各メソッドでwith文を使用）
        pass
    
    def _ensure_database_directory(self):
        """データベースファイルのディレクトリが存在することを確認"""
        db_dir = Path(self.db_path).parent
        if db_dir != Path('.'):
            db_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        データベース接続を取得（エラーハンドリング付き）
        
        Returns:
            sqlite3.Connection: データベース接続
            
        Raises:
            sqlite3.Error: データベース接続エラー
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                conn.execute("PRAGMA foreign_keys = ON")
                return conn
            except sqlite3.Error as e:
                self.logger.warning(f"データベース接続試行 {attempt + 1}/{max_retries} 失敗: {e}")
                if attempt == max_retries - 1:
                    self.logger.error(f"データベース接続に失敗しました: {e}")
                    raise
        
    def create_table(self) -> None:
        """
        ベクトルデータ保存用のテーブルを作成
        
        Raises:
            sqlite3.Error: テーブル作成エラー
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS document_vectors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_key TEXT UNIQUE NOT NULL,
                        vector_data BLOB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # インデックスを作成（存在しない場合のみ）
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_document_key 
                    ON document_vectors(document_key)
                """)
                
                conn.commit()
                self.logger.info("データベーステーブルを正常に作成/確認しました")
                
        except sqlite3.Error as e:
            self.logger.error(f"テーブル作成エラー: {e}")
            raise 
   
    def store_vector(self, key: str, vector: np.ndarray) -> None:
        """
        ベクトルデータをデータベースに保存
        
        Args:
            key (str): ドキュメントキー（拡張子なしのファイル名）
            vector (np.ndarray): ベクトルデータ
            
        Raises:
            sqlite3.Error: データベース操作エラー
            ValueError: 無効な入力データ
        """
        if not key or not key.strip():
            raise ValueError("ドキュメントキーは空にできません")
        
        if vector is None or vector.size == 0:
            raise ValueError("ベクトルデータは空にできません")
        
        try:
            # ベクトルをpickle形式でシリアライズ
            vector_blob = pickle.dumps(vector.astype(np.float32))
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # 既存レコードの確認
                cursor.execute(
                    "SELECT id FROM document_vectors WHERE document_key = ?",
                    (key,)
                )
                existing = cursor.fetchone()
                
                if existing:
                    # 既存レコードを更新
                    cursor.execute("""
                        UPDATE document_vectors 
                        SET vector_data = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE document_key = ?
                    """, (vector_blob, key))
                    self.logger.info(f"ベクトルデータを更新しました: {key}")
                else:
                    # 新規レコードを挿入
                    cursor.execute("""
                        INSERT INTO document_vectors (document_key, vector_data)
                        VALUES (?, ?)
                    """, (key, vector_blob))
                    self.logger.info(f"新しいベクトルデータを保存しました: {key}")
                
                conn.commit()
                
        except sqlite3.Error as e:
            self.logger.error(f"ベクトル保存エラー (key: {key}): {e}")
            raise
        except Exception as e:
            self.logger.error(f"ベクトルシリアライズエラー (key: {key}): {e}")
            raise ValueError(f"ベクトルデータの処理に失敗しました: {e}")
    
    def get_vector(self, key: str) -> Optional[np.ndarray]:
        """
        指定されたキーのベクトルデータを取得
        
        Args:
            key (str): ドキュメントキー
            
        Returns:
            Optional[np.ndarray]: ベクトルデータ（存在しない場合はNone）
            
        Raises:
            sqlite3.Error: データベース操作エラー
        """
        if not key or not key.strip():
            self.logger.warning("空のキーが指定されました")
            return None
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT vector_data FROM document_vectors WHERE document_key = ?",
                    (key,)
                )
                result = cursor.fetchone()
                
                if result:
                    # pickleからベクトルをデシリアライズ
                    vector = pickle.loads(result[0])
                    self.logger.debug(f"ベクトルデータを取得しました: {key}")
                    return vector
                else:
                    self.logger.debug(f"ベクトルデータが見つかりません: {key}")
                    return None
                    
        except sqlite3.Error as e:
            self.logger.error(f"ベクトル取得エラー (key: {key}): {e}")
            raise
        except Exception as e:
            self.logger.error(f"ベクトルデシリアライズエラー (key: {key}): {e}")
            raise ValueError(f"ベクトルデータの読み込みに失敗しました: {e}")
    
    def get_all_vectors(self) -> Dict[str, np.ndarray]:
        """
        すべてのベクトルデータを取得
        
        Returns:
            Dict[str, np.ndarray]: キーとベクトルのマッピング
            
        Raises:
            sqlite3.Error: データベース操作エラー
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT document_key, vector_data FROM document_vectors ORDER BY document_key"
                )
                results = cursor.fetchall()
                
                vectors = {}
                for key, vector_blob in results:
                    try:
                        vector = pickle.loads(vector_blob)
                        vectors[key] = vector
                    except Exception as e:
                        self.logger.warning(f"ベクトルデシリアライズ失敗 (key: {key}): {e}")
                        continue
                
                self.logger.info(f"{len(vectors)}個のベクトルデータを取得しました")
                return vectors
                
        except sqlite3.Error as e:
            self.logger.error(f"全ベクトル取得エラー: {e}")
            raise  
  
    def update_vector(self, key: str, vector: np.ndarray) -> None:
        """
        既存のベクトルデータを更新
        
        Args:
            key (str): ドキュメントキー
            vector (np.ndarray): 新しいベクトルデータ
            
        Raises:
            sqlite3.Error: データベース操作エラー
            ValueError: 無効な入力データまたはキーが存在しない
        """
        if not key or not key.strip():
            raise ValueError("ドキュメントキーは空にできません")
        
        if vector is None or vector.size == 0:
            raise ValueError("ベクトルデータは空にできません")
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # キーの存在確認
                cursor.execute(
                    "SELECT id FROM document_vectors WHERE document_key = ?",
                    (key,)
                )
                if not cursor.fetchone():
                    raise ValueError(f"指定されたキーが存在しません: {key}")
                
                # ベクトルをpickle形式でシリアライズ
                vector_blob = pickle.dumps(vector.astype(np.float32))
                
                # レコードを更新
                cursor.execute("""
                    UPDATE document_vectors 
                    SET vector_data = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE document_key = ?
                """, (vector_blob, key))
                
                conn.commit()
                self.logger.info(f"ベクトルデータを更新しました: {key}")
                
        except sqlite3.Error as e:
            self.logger.error(f"ベクトル更新エラー (key: {key}): {e}")
            raise
        except Exception as e:
            self.logger.error(f"ベクトル更新処理エラー (key: {key}): {e}")
            raise
    
    def delete_vector(self, key: str) -> bool:
        """
        指定されたキーのベクトルデータを削除
        
        Args:
            key (str): ドキュメントキー
            
        Returns:
            bool: 削除が成功した場合True、キーが存在しなかった場合False
            
        Raises:
            sqlite3.Error: データベース操作エラー
        """
        if not key or not key.strip():
            self.logger.warning("空のキーが指定されました")
            return False
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM document_vectors WHERE document_key = ?",
                    (key,)
                )
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    self.logger.info(f"ベクトルデータを削除しました: {key}")
                    return True
                else:
                    self.logger.debug(f"削除対象のキーが見つかりません: {key}")
                    return False
                    
        except sqlite3.Error as e:
            self.logger.error(f"ベクトル削除エラー (key: {key}): {e}")
            raise
    
    def get_vector_count(self) -> int:
        """
        保存されているベクトルの総数を取得
        
        Returns:
            int: ベクトルの総数
            
        Raises:
            sqlite3.Error: データベース操作エラー
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM document_vectors")
                count = cursor.fetchone()[0]
                self.logger.debug(f"ベクトル総数: {count}")
                return count
                
        except sqlite3.Error as e:
            self.logger.error(f"ベクトル数取得エラー: {e}")
            raise
    
    def get_all_keys(self) -> List[str]:
        """
        すべてのドキュメントキーを取得
        
        Returns:
            List[str]: ドキュメントキーのリスト
            
        Raises:
            sqlite3.Error: データベース操作エラー
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT document_key FROM document_vectors ORDER BY document_key"
                )
                keys = [row[0] for row in cursor.fetchall()]
                self.logger.debug(f"{len(keys)}個のキーを取得しました")
                return keys
                
        except sqlite3.Error as e:
            self.logger.error(f"キー一覧取得エラー: {e}")
            raise