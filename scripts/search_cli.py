#!/usr/bin/env python3
"""
検索用CLIツール
Search CLI tool for Japanese HTML Vector Search System
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Optional

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.query_engine import QueryEngine
from src.logger import setup_logger
from config.settings import DATABASE_CONFIG, SEARCH_CONFIG


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    ログ設定を初期化
    Initialize logging configuration
    
    Args:
        log_level: ログレベル
    
    Returns:
        設定されたロガー
    """
    return setup_logger('search_cli', log_level)


def parse_arguments() -> argparse.Namespace:
    """
    コマンドライン引数を解析
    Parse command line arguments
    
    Returns:
        解析された引数
    """
    parser = argparse.ArgumentParser(
        description='日本語HTMLベクトル検索CLI / Japanese HTML Vector Search CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例 / Examples:
  %(prog)s --text "徳川家康"                    # テキストクエリで検索
  %(prog)s --document "page-bushou-徳川家康"    # ドキュメントキーで類似検索
  %(prog)s --text "戦国時代" --top-k 10         # 上位10件を取得
  %(prog)s --interactive                        # インタラクティブモード
        """
    )
    
    # 検索モード
    search_group = parser.add_mutually_exclusive_group(required=False)
    search_group.add_argument(
        '--text',
        type=str,
        help='検索するテキストクエリ'
    )
    
    search_group.add_argument(
        '--document',
        type=str,
        help='類似検索の基準となるドキュメントキー'
    )
    
    search_group.add_argument(
        '--interactive',
        action='store_true',
        help='インタラクティブモードで起動'
    )
    
    # 検索オプション
    parser.add_argument(
        '--top-k',
        type=int,
        default=SEARCH_CONFIG['default_top_k'],
        help=f'取得する結果数 (デフォルト: {SEARCH_CONFIG["default_top_k"]})'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=SEARCH_CONFIG['similarity_threshold'],
        help=f'類似度の閾値 (デフォルト: {SEARCH_CONFIG["similarity_threshold"]})'
    )
    
    parser.add_argument(
        '--db-path',
        type=str,
        default=str(DATABASE_CONFIG['db_path']),
        help=f'データベースファイルパス (デフォルト: {DATABASE_CONFIG["db_path"]})'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='WARNING',
        help='ログレベル (デフォルト: WARNING)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['simple', 'detailed', 'json'],
        default='simple',
        help='出力フォーマット (デフォルト: simple)'
    )
    
    return parser.parse_args()


def format_results(results: List[Tuple[str, float]], format_type: str = 'simple') -> str:
    """
    検索結果をフォーマット
    Format search results
    
    Args:
        results: 検索結果のリスト (ドキュメントキー, 類似度スコア)
        format_type: 出力フォーマット
    
    Returns:
        フォーマットされた結果文字列
    """
    if not results:
        return "検索結果が見つかりませんでした。"
    
    if format_type == 'json':
        import json
        result_data = {
            "total_results": len(results),
            "results": [
                {
                    "rank": i + 1,
                    "document_key": key, 
                    "similarity_score": float(score),
                    "similarity_percentage": round(float(score) * 100, 1)
                }
                for i, (key, score) in enumerate(results)
            ]
        }
        return json.dumps(result_data, ensure_ascii=False, indent=2)
    
    elif format_type == 'detailed':
        output = f"検索結果 ({len(results)}件):\n"
        output += "=" * 60 + "\n"
        for i, (key, score) in enumerate(results, 1):
            output += f"{i:2d}. ドキュメント: {key}\n"
            output += f"    類似度スコア: {score:.4f} ({score*100:.1f}%)\n"
            # スコアバーの表示
            bar_length = 20
            filled_length = int(bar_length * score)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            output += f"    類似度バー: [{bar}]\n"
            output += "-" * 40 + "\n"
        return output
    
    else:  # simple
        output = f"検索結果 ({len(results)}件):\n"
        for i, (key, score) in enumerate(results, 1):
            output += f"{i:2d}. {key} (類似度: {score:.4f})\n"
        return output


def execute_search(query_engine: QueryEngine, query_type: str, query: str, 
                  top_k: int, threshold: float, format_type: str, logger: logging.Logger) -> None:
    """
    検索を実行して結果を表示
    Execute search and display results
    
    Args:
        query_engine: クエリエンジン
        query_type: クエリタイプ ('text' または 'document')
        query: クエリ文字列
        top_k: 取得する結果数
        threshold: 類似度閾値
        format_type: 出力フォーマット
        logger: ロガー
    """
    try:
        logger.debug(f"検索実行: タイプ={query_type}, クエリ={query}")
        
        if query_type == 'text':
            results = query_engine.search_by_text(query, top_k=top_k, threshold=threshold)
        elif query_type == 'document':
            results = query_engine.search_by_document_key(query, top_k=top_k, threshold=threshold)
        else:
            logger.error(f"不正なクエリタイプ: {query_type}")
            print(f"エラー: 不正なクエリタイプです: {query_type}")
            return
        
        # 結果の表示
        formatted_results = format_results(results, format_type)
        print(formatted_results)
        
    except Exception as e:
        logger.error(f"検索エラー: {e}")
        print(f"検索中にエラーが発生しました: {e}")


def interactive_mode(query_engine: QueryEngine, logger: logging.Logger) -> None:
    """
    インタラクティブモードの実行
    Run interactive mode
    
    Args:
        query_engine: クエリエンジン
        logger: ロガー
    """
    print("=== 日本語HTMLベクトル検索 インタラクティブモード ===")
    print("コマンド:")
    print("  text <クエリ>      - テキストクエリで検索")
    print("  doc <ドキュメントキー> - ドキュメント類似検索")
    print("  set top-k <数値>   - 取得結果数を設定")
    print("  set threshold <数値> - 類似度閾値を設定")
    print("  set format <形式>  - 出力フォーマットを設定 (simple/detailed/json)")
    print("  help              - ヘルプを表示")
    print("  quit              - 終了")
    print()
    
    # デフォルト設定
    top_k = SEARCH_CONFIG['default_top_k']
    threshold = SEARCH_CONFIG['similarity_threshold']
    format_type = 'simple'
    
    while True:
        try:
            user_input = input("検索> ").strip()
            
            if not user_input:
                continue
            
            parts = user_input.split(None, 1)
            command = parts[0].lower()
            
            if command == 'quit' or command == 'exit':
                print("検索を終了します。")
                break
            
            elif command == 'help':
                print("利用可能なコマンド:")
                print("  text <クエリ>      - テキストクエリで検索")
                print("                       例: text 徳川家康")
                print("  doc <ドキュメントキー> - ドキュメント類似検索")
                print("                       例: doc page-bushou-徳川家康")
                print("  set top-k <数値>   - 取得結果数を設定 (現在: {})".format(top_k))
                print("  set threshold <数値> - 類似度閾値を設定 (現在: {})".format(threshold))
                print("  set format <形式>  - 出力フォーマットを設定 (現在: {})".format(format_type))
                print("                       形式: simple, detailed, json")
                print("  help              - このヘルプを表示")
                print("  quit              - 終了")
            
            elif command == 'text':
                if len(parts) < 2:
                    print("使用法: text <検索クエリ>")
                    continue
                query = parts[1]
                execute_search(query_engine, 'text', query, top_k, threshold, format_type, logger)
            
            elif command == 'doc':
                if len(parts) < 2:
                    print("使用法: doc <ドキュメントキー>")
                    continue
                doc_key = parts[1]
                execute_search(query_engine, 'document', doc_key, top_k, threshold, format_type, logger)
            
            elif command == 'set':
                if len(parts) < 2:
                    print("使用法: set <パラメータ> <値>")
                    continue
                
                set_parts = parts[1].split(None, 1)
                if len(set_parts) < 2:
                    print("使用法: set <パラメータ> <値>")
                    continue
                
                param = set_parts[0].lower()
                value = set_parts[1]
                
                if param == 'top-k':
                    try:
                        top_k = int(value)
                        print(f"取得結果数を {top_k} に設定しました")
                    except ValueError:
                        print("top-kは整数で指定してください")
                
                elif param == 'threshold':
                    try:
                        threshold = float(value)
                        print(f"類似度閾値を {threshold} に設定しました")
                    except ValueError:
                        print("thresholdは数値で指定してください")
                
                elif param == 'format':
                    if value in ['simple', 'detailed', 'json']:
                        format_type = value
                        print(f"出力フォーマットを {format_type} に設定しました")
                    else:
                        print("formatは simple, detailed, json のいずれかを指定してください")
                
                else:
                    print(f"不明なパラメータ: {param}")
            
            else:
                print(f"不明なコマンド: {command}")
                print("'help' でヘルプを表示します")
        
        except KeyboardInterrupt:
            print("\n\n検索を終了します。")
            break
        except EOFError:
            print("\n検索を終了します。")
            break
        except Exception as e:
            logger.error(f"インタラクティブモードでエラー: {e}")
            print(f"エラーが発生しました: {e}")
            print("続行するには何かキーを押してください...")


def main():
    """
    メイン処理
    Main processing function
    """
    args = parse_arguments()
    logger = setup_logging(args.log_level)
    
    # データベースファイルの存在確認
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"エラー: データベースファイルが見つかりません: {args.db_path}")
        print("先にバッチ処理を実行してベクトルデータを作成してください。")
        sys.exit(1)
    
    try:
        # QueryEngineの初期化
        query_engine = QueryEngine(db_path=args.db_path)
        
        # 検索モードの判定と実行
        if args.interactive:
            interactive_mode(query_engine, logger)
        elif args.text:
            execute_search(query_engine, 'text', args.text, args.top_k, args.threshold, args.format, logger)
        elif args.document:
            execute_search(query_engine, 'document', args.document, args.top_k, args.threshold, args.format, logger)
        else:
            print("検索クエリまたはモードを指定してください。")
            print("使用法: --text <クエリ>, --document <キー>, または --interactive")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}")
        print(f"エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()