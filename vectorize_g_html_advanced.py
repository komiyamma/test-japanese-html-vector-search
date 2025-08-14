#!/usr/bin/env python3
"""
G:\htmlフォルダのHTMLファイルをベクトル化する高度なサンプルスクリプト
Advanced sample script to vectorize HTML files from G:\html folder with custom options
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.batch_processor import BatchProcessor
from src.logger import setup_logger
from src.database_manager import DatabaseManager
from src.query_engine import QueryEngine
from config.settings import DATABASE_CONFIG, EMBEDDING_CONFIG


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    ログ設定を初期化
    Initialize logging configuration
    
    Args:
        log_level: ログレベル
        log_file: ログファイルパス
    
    Returns:
        設定されたロガー
    """
    logger = setup_logger('g_html_vectorizer_advanced', log_level)
    
    if log_file:
        # ファイルハンドラーを追加
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        description='G:\htmlフォルダのHTMLファイルをベクトル化する高度なサンプル',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例 / Examples:
  %(prog)s                                    # デフォルト設定で実行
  %(prog)s --force                            # 既存データを強制更新
  %(prog)s --pattern "page-*.html"            # 特定パターンのファイルのみ処理
  %(prog)s --db-path "my_vectors.db"          # カスタムデータベースファイル
  %(prog)s --log-level DEBUG --log-file vectorize.log  # デバッグログをファイル出力
  %(prog)s --test-search "戦国時代"            # ベクトル化後にテスト検索を実行
        """
    )
    
    parser.add_argument(
        '--source-dir',
        type=str,
        default=r"G:\html",
        help='処理対象のHTMLファイルディレクトリ (デフォルト: G:\html)'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default="*.html",
        help='処理対象ファイルのパターン (デフォルト: *.html)'
    )
    
    parser.add_argument(
        '--db-path',
        type=str,
        default="g_html_vectors.db",
        help='データベースファイルパス (デフォルト: g_html_vectors.db)'
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
    
    parser.add_argument(
        '--test-search',
        type=str,
        help='ベクトル化完了後にテスト検索を実行するクエリテキスト'
    )
    
    parser.add_argument(
        '--show-stats',
        action='store_true',
        help='処理完了後にデータベース統計情報を表示'
    )
    
    return parser.parse_args()


def show_database_stats(db_path: str, logger: logging.Logger):
    """
    データベース統計情報を表示
    Show database statistics
    
    Args:
        db_path: データベースファイルパス
        logger: ロガー
    """
    try:
        db_manager = DatabaseManager(db_path)
        vectors = db_manager.get_all_vectors()
        
        logger.info("=== データベース統計情報 ===")
        logger.info(f"総ベクトル数: {len(vectors)}")
        
        if vectors:
            # ベクトル次元数を取得
            first_vector = next(iter(vectors.values()))
            logger.info(f"ベクトル次元数: {len(first_vector)}")
            
            # ファイル名の例を表示
            keys = list(vectors.keys())
            logger.info("登録されているドキュメント（最初の10件）:")
            for i, key in enumerate(keys[:10]):
                logger.info(f"  {i+1}. {key}")
            if len(keys) > 10:
                logger.info(f"  ... 他 {len(keys) - 10} 件")
        
        # データベースファイルサイズ
        db_file = Path(db_path)
        if db_file.exists():
            size_mb = db_file.stat().st_size / (1024 * 1024)
            logger.info(f"データベースファイルサイズ: {size_mb:.2f} MB")
    
    except Exception as e:
        logger.error(f"統計情報の取得でエラーが発生しました: {e}")


def run_test_search(db_path: str, query_text: str, logger: logging.Logger):
    """
    テスト検索を実行
    Run test search
    
    Args:
        db_path: データベースファイルパス
        query_text: 検索クエリ
        logger: ロガー
    """
    try:
        logger.info(f"=== テスト検索: '{query_text}' ===")
        
        query_engine = QueryEngine(db_path)
        results = query_engine.search_by_text(query_text, top_k=5)
        
        if results:
            logger.info(f"検索結果 ({len(results)}件):")
            for i, (doc_key, similarity) in enumerate(results, 1):
                logger.info(f"  {i}. {doc_key} (類似度: {similarity:.4f})")
        else:
            logger.info("検索結果が見つかりませんでした")
    
    except Exception as e:
        logger.error(f"テスト検索でエラーが発生しました: {e}")


def main():
    """
    メイン処理
    Main processing function
    """
    # コマンドライン引数の解析
    args = parse_arguments()
    
    # ログ設定
    logger = setup_logging(args.log_level, args.log_file)
    
    logger.info("=== G:\htmlフォルダ HTMLファイルベクトル化処理（高度版）を開始 ===")
    
    # 設定情報の表示
    logger.info(f"処理ディレクトリ: {args.source_dir}")
    logger.info(f"ファイルパターン: {args.pattern}")
    logger.info(f"データベースパス: {args.db_path}")
    logger.info(f"使用モデル: {args.model_name}")
    logger.info(f"バッチサイズ: {args.batch_size}")
    logger.info(f"強制更新: {args.force}")
    logger.info(f"ドライラン: {args.dry_run}")
    
    # ディレクトリの存在確認
    directory_path = Path(args.source_dir)
    if not directory_path.exists():
        logger.error(f"指定されたディレクトリが存在しません: {args.source_dir}")
        logger.error("G:\htmlフォルダが存在することを確認してください")
        sys.exit(1)
    
    if not directory_path.is_dir():
        logger.error(f"指定されたパスはディレクトリではありません: {args.source_dir}")
        sys.exit(1)
    
    # HTMLファイルの存在確認
    html_files = list(directory_path.glob(args.pattern))
    if not html_files:
        logger.warning(f"指定されたパターンに一致するファイルが見つかりません: {args.pattern}")
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
            db_path=args.db_path,
            model_name=args.model_name,
            batch_size=args.batch_size
        )
        
        if args.dry_run:
            logger.info("ドライランモード: 処理対象ファイルを表示します")
            files = processor.discover_html_files(args.source_dir, args.pattern)
            logger.info(f"処理対象ファイル数: {len(files)}")
            for file_path in files:
                logger.info(f"  - {file_path}")
            return 0
        
        logger.info("バッチ処理を開始します...")
        
        # バッチ処理の実行
        success_count, error_count = processor.process_directory(
            directory=args.source_dir,
            file_pattern=args.pattern,
            force_update=args.force
        )
        
        logger.info("=== バッチ処理完了 ===")
        logger.info(f"成功: {success_count}ファイル")
        logger.info(f"エラー: {error_count}ファイル")
        logger.info(f"データベースファイル: {Path(args.db_path).absolute()}")
        
        # 統計情報の表示
        if args.show_stats and success_count > 0:
            show_database_stats(args.db_path, logger)
        
        # テスト検索の実行
        if args.test_search and success_count > 0:
            run_test_search(args.db_path, args.test_search, logger)
        
        if success_count > 0:
            logger.info("ベクトル化が正常に完了しました！")
            logger.info("検索を実行するには以下のコマンドを使用してください:")
            logger.info(f"  python scripts/search_cli.py --db-path {args.db_path}")
        
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