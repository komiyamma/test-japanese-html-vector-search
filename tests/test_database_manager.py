"""
DatabaseManagerクラスの単体テスト
"""

import unittest
import tempfile
import os
import numpy as np
import sqlite3
from pathlib import Path
import logging

from src.database_manager import DatabaseManager


class TestDatabaseManager(unittest.TestCase):
    """DatabaseManagerクラスのテストケース"""
    
    def setUp(self):
        """各テストの前に実行される初期化処理"""
        # 一時ディレクトリにテスト用データベースを作成
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_vectors.db")
        self.db_manager = DatabaseManager(self.test_db_path)
        
        # テスト用のベクトルデータ
        self.test_vector1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        self.test_vector2 = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
        self.test_key1 = "page-bushou-羽柴秀吉"
        self.test_key2 = "page-bushou-徳川家康"
        
        # ログレベルを設定（テスト中のログ出力を制御）
        logging.getLogger().setLevel(logging.WARNING)
    
    def tearDown(self):
        """各テストの後に実行されるクリーンアップ処理"""
        # DatabaseManagerインスタンスを削除してリソースを解放
        del self.db_manager
        
        # 少し待機してファイルロックが解除されるのを待つ
        import time
        time.sleep(0.1)
        
        # テスト用データベースファイルを削除
        try:
            if os.path.exists(self.test_db_path):
                os.remove(self.test_db_path)
        except PermissionError:
            # Windowsでファイルロックが残っている場合は警告を出すが続行
            import warnings
            warnings.warn(f"テストデータベースファイルを削除できませんでした: {self.test_db_path}")
        
        # 一時ディレクトリを削除
        try:
            os.rmdir(self.temp_dir)
        except OSError:
            # ディレクトリが空でない場合は警告を出すが続行
            import warnings
            warnings.warn(f"一時ディレクトリを削除できませんでした: {self.temp_dir}")
    
    def test_create_table(self):
        """テーブル作成のテスト"""
        # テーブルが正常に作成されることを確認
        with sqlite3.connect(self.test_db_path) as conn:
            cursor = conn.cursor()
            
            # テーブルの存在確認
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='document_vectors'
            """)
            self.assertIsNotNone(cursor.fetchone())
            
            # インデックスの存在確認
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND name='idx_document_key'
            """)
            self.assertIsNotNone(cursor.fetchone())
    
    def test_store_vector_new(self):
        """新規ベクトル保存のテスト"""
        # 新しいベクトルを保存
        self.db_manager.store_vector(self.test_key1, self.test_vector1)
        
        # データベースに正しく保存されたことを確認
        stored_vector = self.db_manager.get_vector(self.test_key1)
        self.assertIsNotNone(stored_vector)
        np.testing.assert_array_almost_equal(stored_vector, self.test_vector1)
    
    def test_store_vector_update(self):
        """既存ベクトル更新のテスト"""
        # 最初のベクトルを保存
        self.db_manager.store_vector(self.test_key1, self.test_vector1)
        
        # 同じキーで異なるベクトルを保存（更新）
        self.db_manager.store_vector(self.test_key1, self.test_vector2)
        
        # 更新されたベクトルが取得できることを確認
        stored_vector = self.db_manager.get_vector(self.test_key1)
        np.testing.assert_array_almost_equal(stored_vector, self.test_vector2)
        
        # レコード数が1つのままであることを確認（重複作成されていない）
        self.assertEqual(self.db_manager.get_vector_count(), 1)
    
    def test_store_vector_invalid_input(self):
        """無効な入力でのベクトル保存テスト"""
        # 空のキーでエラーが発生することを確認
        with self.assertRaises(ValueError):
            self.db_manager.store_vector("", self.test_vector1)
        
        with self.assertRaises(ValueError):
            self.db_manager.store_vector("   ", self.test_vector1)
        
        # 空のベクトルでエラーが発生することを確認
        with self.assertRaises(ValueError):
            self.db_manager.store_vector(self.test_key1, np.array([]))
        
        with self.assertRaises(ValueError):
            self.db_manager.store_vector(self.test_key1, None)
    
    def test_get_vector_existing(self):
        """既存ベクトル取得のテスト"""
        # ベクトルを保存
        self.db_manager.store_vector(self.test_key1, self.test_vector1)
        
        # 保存したベクトルを取得
        retrieved_vector = self.db_manager.get_vector(self.test_key1)
        
        # 正しいベクトルが取得できることを確認
        self.assertIsNotNone(retrieved_vector)
        np.testing.assert_array_almost_equal(retrieved_vector, self.test_vector1)
    
    def test_get_vector_nonexistent(self):
        """存在しないベクトル取得のテスト"""
        # 存在しないキーでNoneが返されることを確認
        result = self.db_manager.get_vector("nonexistent-key")
        self.assertIsNone(result)
    
    def test_get_vector_invalid_key(self):
        """無効なキーでのベクトル取得テスト"""
        # 空のキーでNoneが返されることを確認
        result = self.db_manager.get_vector("")
        self.assertIsNone(result)
        
        result = self.db_manager.get_vector("   ")
        self.assertIsNone(result)    

    def test_get_all_vectors(self):
        """全ベクトル取得のテスト"""
        # 複数のベクトルを保存
        self.db_manager.store_vector(self.test_key1, self.test_vector1)
        self.db_manager.store_vector(self.test_key2, self.test_vector2)
        
        # 全ベクトルを取得
        all_vectors = self.db_manager.get_all_vectors()
        
        # 正しい数のベクトルが取得できることを確認
        self.assertEqual(len(all_vectors), 2)
        
        # 各ベクトルが正しく取得できることを確認
        self.assertIn(self.test_key1, all_vectors)
        self.assertIn(self.test_key2, all_vectors)
        np.testing.assert_array_almost_equal(all_vectors[self.test_key1], self.test_vector1)
        np.testing.assert_array_almost_equal(all_vectors[self.test_key2], self.test_vector2)
    
    def test_get_all_vectors_empty(self):
        """空のデータベースでの全ベクトル取得テスト"""
        all_vectors = self.db_manager.get_all_vectors()
        self.assertEqual(len(all_vectors), 0)
        self.assertIsInstance(all_vectors, dict)
    
    def test_update_vector_existing(self):
        """既存ベクトル更新のテスト"""
        # 最初のベクトルを保存
        self.db_manager.store_vector(self.test_key1, self.test_vector1)
        
        # ベクトルを更新
        self.db_manager.update_vector(self.test_key1, self.test_vector2)
        
        # 更新されたベクトルが取得できることを確認
        updated_vector = self.db_manager.get_vector(self.test_key1)
        np.testing.assert_array_almost_equal(updated_vector, self.test_vector2)
    
    def test_update_vector_nonexistent(self):
        """存在しないキーでの更新テスト"""
        # 存在しないキーで更新を試行するとエラーが発生することを確認
        with self.assertRaises(ValueError):
            self.db_manager.update_vector("nonexistent-key", self.test_vector1)
    
    def test_update_vector_invalid_input(self):
        """無効な入力での更新テスト"""
        # ベクトルを保存
        self.db_manager.store_vector(self.test_key1, self.test_vector1)
        
        # 空のキーでエラーが発生することを確認
        with self.assertRaises(ValueError):
            self.db_manager.update_vector("", self.test_vector2)
        
        # 空のベクトルでエラーが発生することを確認
        with self.assertRaises(ValueError):
            self.db_manager.update_vector(self.test_key1, np.array([]))
    
    def test_delete_vector_existing(self):
        """既存ベクトル削除のテスト"""
        # ベクトルを保存
        self.db_manager.store_vector(self.test_key1, self.test_vector1)
        
        # ベクトルを削除
        result = self.db_manager.delete_vector(self.test_key1)
        self.assertTrue(result)
        
        # 削除されたことを確認
        deleted_vector = self.db_manager.get_vector(self.test_key1)
        self.assertIsNone(deleted_vector)
    
    def test_delete_vector_nonexistent(self):
        """存在しないベクトル削除のテスト"""
        # 存在しないキーで削除を試行
        result = self.db_manager.delete_vector("nonexistent-key")
        self.assertFalse(result)
    
    def test_delete_vector_invalid_key(self):
        """無効なキーでの削除テスト"""
        # 空のキーで削除を試行
        result = self.db_manager.delete_vector("")
        self.assertFalse(result)
        
        result = self.db_manager.delete_vector("   ")
        self.assertFalse(result)
    
    def test_get_vector_count(self):
        """ベクトル数取得のテスト"""
        # 初期状態では0
        self.assertEqual(self.db_manager.get_vector_count(), 0)
        
        # 1つ追加
        self.db_manager.store_vector(self.test_key1, self.test_vector1)
        self.assertEqual(self.db_manager.get_vector_count(), 1)
        
        # もう1つ追加
        self.db_manager.store_vector(self.test_key2, self.test_vector2)
        self.assertEqual(self.db_manager.get_vector_count(), 2)
        
        # 1つ削除
        self.db_manager.delete_vector(self.test_key1)
        self.assertEqual(self.db_manager.get_vector_count(), 1)
    
    def test_get_all_keys(self):
        """全キー取得のテスト"""
        # 初期状態では空のリスト
        keys = self.db_manager.get_all_keys()
        self.assertEqual(len(keys), 0)
        
        # ベクトルを追加
        self.db_manager.store_vector(self.test_key1, self.test_vector1)
        self.db_manager.store_vector(self.test_key2, self.test_vector2)
        
        # キーが正しく取得できることを確認
        keys = self.db_manager.get_all_keys()
        self.assertEqual(len(keys), 2)
        self.assertIn(self.test_key1, keys)
        self.assertIn(self.test_key2, keys)
        
        # ソートされていることを確認
        self.assertEqual(keys, sorted(keys))
    
    def test_japanese_keys(self):
        """日本語キーの処理テスト"""
        japanese_keys = [
            "page-bushou-羽柴秀吉",
            "page-bushou-徳川家康",
            "page-bushou-武田信玄",
            "page-bushou-織田信長"
        ]
        
        # 日本語キーでベクトルを保存
        for i, key in enumerate(japanese_keys):
            vector = np.array([i, i+1, i+2, i+3], dtype=np.float32)
            self.db_manager.store_vector(key, vector)
        
        # 保存されたキーを取得
        stored_keys = self.db_manager.get_all_keys()
        
        # すべてのキーが正しく保存・取得できることを確認
        self.assertEqual(len(stored_keys), len(japanese_keys))
        for key in japanese_keys:
            self.assertIn(key, stored_keys)
    
    def test_large_vector(self):
        """大きなベクトルの処理テスト"""
        # 384次元のベクトル（実際のモデル出力サイズ）
        large_vector = np.random.rand(384).astype(np.float32)
        
        # 大きなベクトルを保存・取得
        self.db_manager.store_vector(self.test_key1, large_vector)
        retrieved_vector = self.db_manager.get_vector(self.test_key1)
        
        # 正しく保存・取得できることを確認
        np.testing.assert_array_almost_equal(retrieved_vector, large_vector)
    
    def test_database_connection_error_handling(self):
        """データベース接続エラーハンドリングのテスト"""
        # 無効なパスでDatabaseManagerを作成
        invalid_path = "/invalid/path/to/database.db"
        
        # 権限エラーが発生する可能性があるパスでテスト
        # （実際のエラーハンドリングの動作確認）
        try:
            db_manager = DatabaseManager(invalid_path)
            # エラーが発生しない場合は、正常に作成されたことを確認
            self.assertIsNotNone(db_manager)
        except (sqlite3.Error, OSError, PermissionError):
            # 期待されるエラーが発生した場合はテスト成功
            pass


if __name__ == '__main__':
    unittest.main()