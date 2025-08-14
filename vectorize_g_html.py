#!/usr/bin/env python3
"""
G:\htmlフォルダのHTMLファイルをベクトル化するサンプルスクリプト
Sample script to vectorize HTML files from G:\html folder
"""

import sys
import logging
from pathlib import Path
from typing import Optional

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.batch_processor import BatchProcessor
from src.logger import setup_logger
from config.settings import DATABASE_CONFIG, EMBEDDING_CONFIG


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    ログ設定を初期化
    Initialize logging configuration
    
    Args:
        log_level: ログレベル
    
    Returns:
        設定されたロガー
    """
    return setup_logger('g_html_vectorizer', log_level)


def main():
    """
    G:\htmlフォルダのHTMLファイルをベクトル化するメイン処理
    Main processing function to vectorize HTML files from G:\html folder
    """
    # ログ設定
    logger = setup_logging("INFO")
    
    logger.info("=== G:\htmlフォルダ HTMLファイルベクトル化処理を開始 ===")
    
    # 処理対象ディレクトリ
    target_directory = r"G:\html"
    
    # データベースファイルパス（現在のディレクトリに作成）
    db_path = "g_html_vectors.db"
    
    # 設定情報の表示
    logger.info(f"処理ディレクトリ: {target_directory}")
    logger.info(f"データベースパス: {db_path}")
    logger.info(f"使用モデル: {EMBEDDING_CONFIG['model_name']}")
    logger.info(f"バッチサイズ: {EMBEDDING_CONFIG['batch_size']}")
    
    # ディレクトリの存在確認
    directory_path = Path(target_directory)
    if not directory_path.exists():
        logger.error(f"指定されたディレクトリが存在しません: {target_directory}")
        logger.error("G:\htmlフォルダが存在することを確認してください")
        sys.exit(1)
    
    if not directory_path.is_dir():
        logger.error(f"指定されたパスはディレクトリではありません: {target_directory}")
        sys.exit(1)
    
    # HTMLファイルの存在確認
    html_files = list(directory_path.glob("*.html"))
    if not html_files:
        logger.warning(f"指定されたディレクトリにHTMLファイルが見つかりません: {target_directory}")
        logger.info("処理を続行しますが、ファイルが見つからない可能性があります")
    else:
        logger.info(f"発見されたHTMLファイル数: {len(html_files)}")
        # 最初の5ファイルを表示
        for i, file_path in enumerate(html_files[:5]):
            logger.info(f"  - {file_path.name}")
        if len(html_files) > 5:
            logger.info(f"  ... 他 {len(html_files) - 5} ファイル")
    
    try:
        # BatchProcessorの初期化
        logger.info("ベクトル化エンジンを初期化中...")
        processor = BatchProcessor(
            db_path=db_path,
            model_name=EMBEDDING_CONFIG['model_name'],
            batch_size=EMBEDDING_CONFIG['batch_size']
        )
        
        logger.info("バッチ処理を開始します...")
        
        # バッチ処理の実行
        # *.htmlパターンですべてのHTMLファイルを処理
        success_count, error_count = processor.process_directory(
            directory=target_directory,
            file_pattern="*.html",  # すべてのHTMLファイルを対象
            force_update=False  # 既存のベクトルデータがある場合はスキップ
        )
        
        logger.info("=== バッチ処理完了 ===")
        logger.info(f"成功: {success_count}ファイル")
        logger.info(f"エラー: {error_count}ファイル")
        logger.info(f"データベースファイル: {Path(db_path).absolute()}")
        
        if success_count > 0:
            logger.info("ベクトル化が正常に完了しました！")
            logger.info("検索を実行するには以下のコマンドを使用してください:")
            logger.info(f"  python scripts/search_cli.py --db-path {db_path}")
        
        if error_count > 0:
            logger.warning("一部のファイルでエラーが発生しました。詳細はログを確認してください。")
            return 1
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("ユーザーによって処理が中断されました")
        return 130
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}")
        logger.exception("詳細なエラー情報:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)