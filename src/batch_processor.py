"""
BatchProcessor - HTMLファイルのバッチ処理クラス

このモジュールは、ディレクトリ内のHTMLファイルを自動発見し、
バッチでベクトル化処理を実行する機能を提供します。
プログレス表示、エラー処理、処理済みファイルのスキップ機能を含みます。
"""

import os
import glob
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .html_processor import HTMLProcessor
from .vector_embedder import VectorEmbedder
from .database_manager import DatabaseManager


@dataclass
class ProcessingResult:
    """処理結果を格納するデータクラス"""
    total_files: int = 0
    processed_files: int = 0
    skipped_files: int = 0
    error_files: int = 0
    processing_time: float = 0.0
    errors: List[Tuple[str, str]] = None  # (ファイル名, エラーメッセージ)
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class BatchProcessor:
    """
    HTMLファイルのバッチ処理を行うクラス
    
    主な機能:
    - HTMLファイルの自動発見
    - プログレス表示付きバッチ処理
    - エラー処理と継続機能
    - 処理済みファイルのスキップ
    - 処理結果サマリー
    """
    
    def __init__(self, 
                 html_processor: Optional[HTMLProcessor] = None,
                 vector_embedder: Optional[VectorEmbedder] = None,
                 database_manager: Optional[DatabaseManager] = None,
                 base_directory: str = "."):
        """
        BatchProcessorを初期化
        
        Args:
            html_processor: HTMLProcessor インスタンス（Noneの場合は新規作成）
            vector_embedder: VectorEmbedder インスタンス（Noneの場合は新規作成）
            database_manager: DatabaseManager インスタンス（Noneの場合は新規作成）
            base_directory: 処理対象ディレクトリのパス
        """
        self.logger = logging.getLogger(__name__)
        self.base_directory = Path(base_directory)
        
        # コンポーネントの初期化
        self.html_processor = html_processor or HTMLProcessor()
        self.vector_embedder = vector_embedder or VectorEmbedder()
        self.database_manager = database_manager or DatabaseManager()
        
        self.logger.info(f"BatchProcessor初期化完了 - ベースディレクトリ: {self.base_directory}")
    
    def discover_html_files(self, pattern: str = "page-*.html") -> List[str]:
        """
        指定されたパターンでHTMLファイルを自動発見
        
        Args:
            pattern: ファイル検索パターン（デフォルト: "page-*.html"）
            
        Returns:
            List[str]: 発見されたHTMLファイルのパスリスト
        """
        try:
            # globパターンでファイルを検索
            search_pattern = str(self.base_directory / pattern)
            html_files = glob.glob(search_pattern)
            
            # パスを正規化してソート
            html_files = [os.path.abspath(f) for f in html_files]
            html_files.sort()
            
            self.logger.info(f"HTMLファイル発見: {len(html_files)}件 (パターン: {pattern})")
            
            if html_files:
                self.logger.debug("発見されたファイル:")
                for file_path in html_files:
                    self.logger.debug(f"  - {file_path}")
            
            return html_files
            
        except Exception as e:
            self.logger.error(f"HTMLファイル発見エラー: {e}")
            return []
    
    def is_file_processed(self, file_path: str) -> bool:
        """
        ファイルが既に処理済みかどうかを確認
        
        Args:
            file_path: チェックするファイルのパス
            
        Returns:
            bool: 処理済みの場合True
        """
        try:
            file_key = self.html_processor.get_file_key(file_path)
            existing_vector = self.database_manager.get_vector(file_key)
            return existing_vector is not None
            
        except Exception as e:
            self.logger.warning(f"処理済み確認エラー ({file_path}): {e}")
            return False
    
    def process_single_file(self, file_path: str, force_reprocess: bool = False) -> bool:
        """
        単一ファイルを処理
        
        Args:
            file_path: 処理するファイルのパス
            force_reprocess: 既に処理済みでも再処理するかどうか
            
        Returns:
            bool: 処理が成功した場合True
        """
        try:
            file_key = self.html_processor.get_file_key(file_path)
            
            # 処理済みチェック
            if not force_reprocess and self.is_file_processed(file_path):
                self.logger.debug(f"スキップ（処理済み）: {file_path}")
                return False  # スキップを示すためFalse
            
            self.logger.debug(f"処理開始: {file_path}")
            
            # HTMLからテキストを抽出
            text = self.html_processor.extract_text(file_path)
            
            # コンテンツ長の検証
            self.html_processor.validate_content_length(text)
            
            # テキストをベクトル化
            vector = self.vector_embedder.embed_text(text)
            
            # データベースに保存
            self.database_manager.store_vector(file_key, vector)
            
            self.logger.info(f"処理完了: {file_path} -> {file_key}")
            return True
            
        except Exception as e:
            error_msg = f"ファイル処理エラー ({file_path}): {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def display_progress(self, current: int, total: int, file_name: str = "", 
                        start_time: float = None) -> None:
        """
        プログレス情報を表示
        
        Args:
            current: 現在の処理数
            total: 総ファイル数
            file_name: 現在処理中のファイル名
            start_time: 処理開始時刻
        """
        percentage = (current / total) * 100 if total > 0 else 0
        
        # 経過時間と推定残り時間を計算
        elapsed_time = ""
        eta = ""
        if start_time and current > 0:
            elapsed = time.time() - start_time
            elapsed_time = f" | 経過: {elapsed:.1f}秒"
            
            if current < total:
                avg_time_per_file = elapsed / current
                remaining_files = total - current
                estimated_remaining = avg_time_per_file * remaining_files
                eta = f" | 残り: {estimated_remaining:.1f}秒"
        
        # ファイル名を短縮表示
        display_name = file_name
        if len(display_name) > 50:
            display_name = "..." + display_name[-47:]
        
        progress_msg = (
            f"進捗: {current}/{total} ({percentage:.1f}%){elapsed_time}{eta}"
        )
        
        if display_name:
            progress_msg += f" | 処理中: {display_name}"
        
        self.logger.info(progress_msg)
    
    def process_batch(self, 
                     file_pattern: str = "page-*.html",
                     force_reprocess: bool = False,
                     show_progress: bool = True,
                     progress_interval: int = 1) -> ProcessingResult:
        """
        バッチ処理を実行
        
        Args:
            file_pattern: 処理対象ファイルのパターン
            force_reprocess: 既に処理済みでも再処理するかどうか
            show_progress: プログレス表示を行うかどうか
            progress_interval: プログレス表示の間隔（ファイル数）
            
        Returns:
            ProcessingResult: 処理結果
        """
        start_time = time.time()
        result = ProcessingResult()
        
        try:
            self.logger.info("バッチ処理を開始します")
            self.logger.info(f"設定 - パターン: {file_pattern}, 再処理: {force_reprocess}")
            
            # HTMLファイルを発見
            html_files = self.discover_html_files(file_pattern)
            result.total_files = len(html_files)
            
            if result.total_files == 0:
                self.logger.warning("処理対象のHTMLファイルが見つかりませんでした")
                return result
            
            self.logger.info(f"処理対象ファイル: {result.total_files}件")
            
            # 各ファイルを処理
            for i, file_path in enumerate(html_files, 1):
                file_name = os.path.basename(file_path)
                
                try:
                    # プログレス表示
                    if show_progress and (i % progress_interval == 0 or i == 1 or i == result.total_files):
                        self.display_progress(i, result.total_files, file_name, start_time)
                    
                    # ファイル処理
                    processed = self.process_single_file(file_path, force_reprocess)
                    
                    if processed:
                        result.processed_files += 1
                    else:
                        result.skipped_files += 1
                        
                except Exception as e:
                    result.error_files += 1
                    error_msg = str(e)
                    result.errors.append((file_name, error_msg))
                    
                    self.logger.error(f"ファイル処理失敗: {file_name} - {error_msg}")
                    
                    # エラーが発生しても処理を継続
                    continue
            
            # 処理時間を記録
            result.processing_time = time.time() - start_time
            
            self.logger.info("バッチ処理が完了しました")
            
        except Exception as e:
            self.logger.error(f"バッチ処理中に予期しないエラー: {e}")
            result.processing_time = time.time() - start_time
            raise
        
        return result
    
    def print_summary(self, result: ProcessingResult) -> None:
        """
        処理結果のサマリーを表示
        
        Args:
            result: 処理結果
        """
        self.logger.info("=" * 60)
        self.logger.info("バッチ処理結果サマリー")
        self.logger.info("=" * 60)
        
        self.logger.info(f"総ファイル数:     {result.total_files}")
        self.logger.info(f"処理済みファイル: {result.processed_files}")
        self.logger.info(f"スキップファイル: {result.skipped_files}")
        self.logger.info(f"エラーファイル:   {result.error_files}")
        self.logger.info(f"処理時間:         {result.processing_time:.2f}秒")
        
        if result.total_files > 0:
            success_rate = ((result.processed_files + result.skipped_files) / result.total_files) * 100
            self.logger.info(f"成功率:           {success_rate:.1f}%")
            
            if result.processing_time > 0 and result.processed_files > 0:
                avg_time = result.processing_time / result.processed_files
                self.logger.info(f"平均処理時間:     {avg_time:.2f}秒/ファイル")
        
        # エラー詳細を表示
        if result.errors:
            self.logger.info("-" * 40)
            self.logger.info("エラー詳細:")
            for file_name, error_msg in result.errors:
                self.logger.error(f"  {file_name}: {error_msg}")
        
        self.logger.info("=" * 60)
    
    def get_processing_statistics(self) -> Dict[str, int]:
        """
        現在のデータベースの統計情報を取得
        
        Returns:
            Dict[str, int]: 統計情報（総ベクトル数など）
        """
        try:
            vector_count = self.database_manager.get_vector_count()
            all_keys = self.database_manager.get_all_keys()
            
            return {
                "total_vectors": vector_count,
                "total_keys": len(all_keys),
                "database_path": str(self.database_manager.db_path)
            }
            
        except Exception as e:
            self.logger.error(f"統計情報取得エラー: {e}")
            return {"error": str(e)}