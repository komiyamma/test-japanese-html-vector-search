#!/usr/bin/env python3
"""
バッチ処理メインスクリプト
Batch processing main script for Japanese HTML Vector Search System
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.batch_processor import BatchProcessor
from src.logger import setup_logger
from config.settings import DATABASE_CONFIG, EMBEDDING_CONFIG, HTML_CONFIG, LOG_CONFIG


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    ログ設定を初期化
    Initialize logging configuration
    
    Args:
        log_level: ログレベル (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: ログファイルパス（指定しない場合はコンソール出力のみ）
    
    Returns:
        設定されたロガー
    """
    logger = setup_logger('batch_processor', log_level)
    
    if log_file:
        # ファイルハンドラーを追加
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        formatter = logging.Formatter(LOG_CONFIG['format'])
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def parse_arguments() -> argparse.Namespace:
    """
    コマンドライン引数を解析
    Parse command line arguments
    
    Returns:
        解析された引数
    """
    parser = argparse.ArgumentParser(
        description='日本語HTMLファイルのバッチベクトル化処理 / Batch vectorization of Japanese HTML files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例 / Examples:
  %(prog)s                          # 現在のディレクトリのHTMLファイルを処理
  %(prog)s -d /path/to/html/files   # 指定ディレクトリのHTMLファイルを処理
  %(prog)s --force                  # 既存のベクトルデータを強制更新
  %(prog)s --log-level DEBUG        # デバッグレベルでログ出力
  %(prog)s --log-file batch.log     # ログをファイルに出力
        """
    )
    
    parser.add_argument(
        '-d', '--directory',
        type=str,
        default='.',
        help='処理対象のHTMLファイルが格納されているディレクトリ (デフォルト: 現在のディレクトリ)'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default=HTML_CONFIG['file_pattern'],
        help=f'処理対象ファイルのパターン (デフォルト: {HTML_CONFIG["file_pattern"]})'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='既存のベクトルデータを強制的に再処理する'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=EMBEDDING_CONFIG['batch_size'],
        help=f'バッチ処理サイズ (デフォルト: {EMBEDDING_CONFIG["batch_size"]})'
    )
    
    parser.add_argument(
        '--db-path',
        type=str,
        default=str(DATABASE_CONFIG['db_path']),
        help=f'データベースファイルパス (デフォルト: {DATABASE_CONFIG["db_path"]})'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default=EMBEDDING_CONFIG['model_name'],
        help=f'使用するベクトル化モデル (デフォルト: {EMBEDDING_CONFIG["model_name"]})'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='ログレベル (デフォルト: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='ログファイルパス（指定しない場合はコンソール出力のみ）'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='実際の処理は行わず、処理対象ファイルのみ表示する'
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace, logger: logging.Logger) -> bool:
    """
    引数の妥当性を検証
    Validate command line arguments
    
    Args:
        args: 解析された引数
        logger: ロガー
    
    Returns:
        引数が妥当な場合True
    """
    # ディレクトリの存在確認
    directory_path = Path(args.directory)
    if not directory_path.exists():
        logger.error(f"指定されたディレクトリが存在しません: {args.directory}")
        return False
    
    if not directory_path.is_dir():
        logger.error(f"指定されたパスはディレクトリではありません: {args.directory}")
        return False
    
    # バッチサイズの妥当性確認
    if args.batch_size <= 0:
        logger.error(f"バッチサイズは正の整数である必要があります: {args.batch_size}")
        return False
    
    # データベースディレクトリの作成
    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # ログファイルディレクトリの作成
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    return True


def main():
    """
    メイン処理
    Main processing function
    """
    # コマンドライン引数の解析
    args = parse_arguments()
    
    # ログ設定
    logger = setup_logging(args.log_level, args.log_file)
    
    logger.info("=== 日本語HTMLベクトル化バッチ処理を開始 ===")
    logger.info(f"処理ディレクトリ: {args.directory}")
    logger.info(f"ファイルパターン: {args.pattern}")
    logger.info(f"データベースパス: {args.db_path}")
    logger.info(f"モデル名: {args.model_name}")
    logger.info(f"バッチサイズ: {args.batch_size}")
    logger.info(f"強制更新: {args.force}")
    logger.info(f"ドライラン: {args.dry_run}")
    
    # 引数の妥当性検証
    if not validate_arguments(args, logger):
        logger.error("引数の検証に失敗しました")
        sys.exit(1)
    
    try:
        # BatchProcessorの初期化
        processor = BatchProcessor(
            db_path=args.db_path,
            model_name=args.model_name,
            batch_size=args.batch_size
        )
        
        # バッチ処理の実行
        if args.dry_run:
            logger.info("ドライランモード: 処理対象ファイルを表示します")
            files = processor.discover_html_files(args.directory, args.pattern)
            logger.info(f"処理対象ファイル数: {len(files)}")
            for file_path in files:
                logger.info(f"  - {file_path}")
        else:
            success_count, error_count = processor.process_directory(
                directory=args.directory,
                file_pattern=args.pattern,
                force_update=args.force
            )
            
            logger.info("=== バッチ処理完了 ===")
            logger.info(f"成功: {success_count}ファイル")
            logger.info(f"エラー: {error_count}ファイル")
            
            if error_count > 0:
                logger.warning("一部のファイルでエラーが発生しました。詳細はログを確認してください。")
                sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("ユーザーによって処理が中断されました")
        sys.exit(130)
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}")
        logger.exception("詳細なエラー情報:")
        sys.exit(1)


if __name__ == "__main__":
    main()