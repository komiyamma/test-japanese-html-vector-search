#!/usr/bin/env python3
"""
日本語HTMLベクトル検索システム メインエントリーポイント
Japanese HTML Vector Search System - Main Entry Point
"""

import argparse
import sys
import subprocess
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.logger import setup_logger
from config.settings import DATABASE_CONFIG, LOG_CONFIG


def parse_arguments() -> argparse.Namespace:
    """
    コマンドライン引数を解析
    Parse command line arguments
    
    Returns:
        解析された引数
    """
    parser = argparse.ArgumentParser(
        description='日本語HTMLベクトル検索システム / Japanese HTML Vector Search System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例 / Examples:
  %(prog)s batch                    # バッチ処理モード
  %(prog)s search                   # 検索モード
  %(prog)s batch --help             # バッチ処理のヘルプ
  %(prog)s search --help            # 検索のヘルプ
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='実行モード')
    
    # バッチ処理モード
    batch_parser = subparsers.add_parser(
        'batch',
        help='HTMLファイルのバッチベクトル化処理'
    )
    batch_parser.add_argument(
        '--help-full',
        action='store_true',
        help='バッチ処理の詳細ヘルプを表示'
    )
    
    # 検索モード
    search_parser = subparsers.add_parser(
        'search',
        help='ベクトル検索の実行'
    )
    search_parser.add_argument(
        '--help-full',
        action='store_true',
        help='検索の詳細ヘルプを表示'
    )
    
    # 共通オプション
    parser.add_argument(
        '--version',
        action='version',
        version='Japanese HTML Vector Search System v1.0.0'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='ログレベル (デフォルト: INFO)'
    )
    
    return parser.parse_args()


def show_system_info(logger):
    """
    システム情報を表示
    Display system information
    
    Args:
        logger: ロガー
    """
    logger.info("=== システム情報 ===")
    logger.info(f"プロジェクトルート: {project_root}")
    logger.info(f"データベースパス: {DATABASE_CONFIG['db_path']}")
    logger.info(f"ログファイル: {LOG_CONFIG['log_file']}")
    
    # データベースファイルの存在確認
    db_path = Path(DATABASE_CONFIG['db_path'])
    if db_path.exists():
        logger.info(f"データベースファイル: 存在 ({db_path.stat().st_size} bytes)")
    else:
        logger.info("データベースファイル: 未作成")
    
    # HTMLファイルの数を確認
    html_files = list(project_root.glob('page-*.html'))
    logger.info(f"HTMLファイル数: {len(html_files)}件")


def run_batch_mode(args, logger):
    """
    バッチ処理モードを実行
    Run batch processing mode
    
    Args:
        args: コマンドライン引数
        logger: ロガー
    """
    if hasattr(args, 'help_full') and args.help_full:
        # バッチ処理の詳細ヘルプを表示
        subprocess.run([sys.executable, 'scripts/batch_process.py', '--help'])
        return
    
    logger.info("バッチ処理モードを開始します")
    
    # バッチ処理スクリプトを実行
    batch_script = project_root / 'scripts' / 'batch_process.py'
    cmd = [sys.executable, str(batch_script), '--log-level', args.log_level]
    
    try:
        result = subprocess.run(cmd, check=True)
        logger.info("バッチ処理が正常に完了しました")
    except subprocess.CalledProcessError as e:
        logger.error(f"バッチ処理でエラーが発生しました: {e}")
        sys.exit(e.returncode)


def run_search_mode(args, logger):
    """
    検索モードを実行
    Run search mode
    
    Args:
        args: コマンドライン引数
        logger: ロガー
    """
    if hasattr(args, 'help_full') and args.help_full:
        # 検索の詳細ヘルプを表示
        subprocess.run([sys.executable, 'scripts/search_cli.py', '--help'])
        return
    
    logger.info("検索モードを開始します")
    
    # 検索CLIスクリプトをインタラクティブモードで実行
    search_script = project_root / 'scripts' / 'search_cli.py'
    cmd = [sys.executable, str(search_script), '--interactive', '--log-level', args.log_level]
    
    try:
        subprocess.run(cmd, check=True)
        logger.info("検索モードを終了しました")
    except subprocess.CalledProcessError as e:
        logger.error(f"検索モードでエラーが発生しました: {e}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        logger.info("ユーザーによって検索モードが中断されました")


def main():
    """
    メイン関数
    Main function
    """
    args = parse_arguments()
    
    # ログ設定
    logger = setup_logger('main', args.log_level)
    
    logger.info("=== 日本語HTMLベクトル検索システム ===")
    logger.info("Japanese HTML Vector Search System v1.0.0")
    
    # システム情報の表示
    show_system_info(logger)
    
    try:
        if args.mode == 'batch':
            run_batch_mode(args, logger)
        elif args.mode == 'search':
            run_search_mode(args, logger)
        else:
            logger.info("実行モードが指定されていません")
            print("使用法: python main.py {batch|search}")
            print("詳細: python main.py --help")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("ユーザーによって処理が中断されました")
        sys.exit(130)
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}")
        logger.exception("詳細なエラー情報:")
        sys.exit(1)
    
    logger.info("システムを終了します")


if __name__ == "__main__":
    main()