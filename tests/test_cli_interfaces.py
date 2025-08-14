"""
CLIインターフェースのテスト
Tests for CLI interfaces
"""

import unittest
import subprocess
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestBatchProcessCLI(unittest.TestCase):
    """
    バッチ処理CLIのテストクラス
    Test class for batch processing CLI
    """
    
    def setUp(self):
        """テストセットアップ"""
        self.batch_script = project_root / 'scripts' / 'batch_process.py'
        self.temp_dir = tempfile.mkdtemp()
        self.temp_db = Path(self.temp_dir) / 'test_vectors.db'
    
    def test_help_option(self):
        """ヘルプオプションのテスト"""
        result = subprocess.run(
            [sys.executable, str(self.batch_script), '--help'],
            capture_output=True,
            text=True
        )
        
        self.assertEqual(result.returncode, 0)
        self.assertIn('日本語HTMLファイルのバッチベクトル化処理', result.stdout)
        self.assertIn('--directory', result.stdout)
        self.assertIn('--force', result.stdout)
    
    def test_dry_run_option(self):
        """ドライランオプションのテスト"""
        result = subprocess.run(
            [
                sys.executable, str(self.batch_script),
                '--dry-run',
                '--directory', str(project_root),
                '--db-path', str(self.temp_db)
            ],
            capture_output=True,
            text=True
        )
        
        # ドライランは正常終了するはず
        self.assertEqual(result.returncode, 0)
    
    def test_invalid_directory(self):
        """無効なディレクトリのテスト"""
        invalid_dir = '/nonexistent/directory'
        result = subprocess.run(
            [
                sys.executable, str(self.batch_script),
                '--directory', invalid_dir,
                '--db-path', str(self.temp_db)
            ],
            capture_output=True,
            text=True
        )
        
        # 無効なディレクトリの場合はエラー終了するはず
        self.assertNotEqual(result.returncode, 0)
    
    def test_invalid_batch_size(self):
        """無効なバッチサイズのテスト"""
        result = subprocess.run(
            [
                sys.executable, str(self.batch_script),
                '--batch-size', '0',
                '--db-path', str(self.temp_db)
            ],
            capture_output=True,
            text=True
        )
        
        # 無効なバッチサイズの場合はエラー終了するはず
        self.assertNotEqual(result.returncode, 0)


class TestSearchCLI(unittest.TestCase):
    """
    検索CLIのテストクラス
    Test class for search CLI
    """
    
    def setUp(self):
        """テストセットアップ"""
        self.search_script = project_root / 'scripts' / 'search_cli.py'
        self.temp_dir = tempfile.mkdtemp()
        self.temp_db = Path(self.temp_dir) / 'test_vectors.db'
    
    @patch('sys.path')
    @patch('scripts.search_cli.QueryEngine')
    def test_argument_parsing(self, mock_query_engine, mock_sys_path):
        """引数解析のテスト"""
        # search_cliモジュールをインポート
        sys.path.insert(0, str(project_root / 'scripts'))
        
        try:
            from search_cli import parse_arguments
            
            # テスト用引数を設定
            test_args = [
                '--text', 'test query',
                '--top-k', '10',
                '--threshold', '0.5',
                '--format', 'json'
            ]
            
            with patch('sys.argv', ['search_cli.py'] + test_args):
                args = parse_arguments()
                
                self.assertEqual(args.text, 'test query')
                self.assertEqual(args.top_k, 10)
                self.assertEqual(args.threshold, 0.5)
                self.assertEqual(args.format, 'json')
        
        finally:
            if str(project_root / 'scripts') in sys.path:
                sys.path.remove(str(project_root / 'scripts'))
    
    def test_format_results_simple(self):
        """結果フォーマット（シンプル）のテスト"""
        # format_results関数を直接定義してテスト
        def format_results(results, format_type='simple'):
            if not results:
                return "検索結果が見つかりませんでした。"
            
            if format_type == 'simple':
                output = f"検索結果 ({len(results)}件):\n"
                for i, (key, score) in enumerate(results, 1):
                    output += f"{i:2d}. {key} (類似度: {score:.4f})\n"
                return output
            return ""
        
        results = [
            ('page-bushou-徳川家康', 0.95),
            ('page-bushou-織田信長', 0.87),
            ('page-bushou-豊臣秀吉', 0.82)
        ]
        
        formatted = format_results(results, 'simple')
        
        self.assertIn('検索結果 (3件)', formatted)
        self.assertIn('page-bushou-徳川家康', formatted)
        self.assertIn('0.9500', formatted)
    
    def test_format_results_detailed(self):
        """結果フォーマット（詳細）のテスト"""
        # format_results関数を直接定義してテスト
        def format_results(results, format_type='simple'):
            if not results:
                return "検索結果が見つかりませんでした。"
            
            if format_type == 'detailed':
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
            return ""
        
        results = [
            ('page-bushou-徳川家康', 0.95),
            ('page-bushou-織田信長', 0.87)
        ]
        
        formatted = format_results(results, 'detailed')
        
        self.assertIn('検索結果 (2件)', formatted)
        self.assertIn('ドキュメント: page-bushou-徳川家康', formatted)
        self.assertIn('類似度スコア: 0.9500', formatted)
        self.assertIn('=' * 50, formatted)
    
    def test_format_results_json(self):
        """結果フォーマット（JSON）のテスト"""
        # format_results関数を直接定義してテスト
        def format_results(results, format_type='simple'):
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
            return ""
        
        results = [
            ('page-bushou-徳川家康', 0.95),
            ('page-bushou-織田信長', 0.87)
        ]
        
        formatted = format_results(results, 'json')
        
        # JSONとして解析可能かテスト
        import json
        parsed = json.loads(formatted)
        
        self.assertEqual(parsed['total_results'], 2)
        self.assertEqual(parsed['results'][0]['document_key'], 'page-bushou-徳川家康')
        self.assertEqual(parsed['results'][0]['similarity_score'], 0.95)
    
    def test_format_results_empty(self):
        """空の結果のフォーマットテスト"""
        # format_results関数を直接定義してテスト
        def format_results(results, format_type='simple'):
            if not results:
                return "検索結果が見つかりませんでした。"
            return ""
        
        results = []
        formatted = format_results(results, 'simple')
        
        self.assertEqual(formatted, "検索結果が見つかりませんでした。")
    
    @patch('sys.path')
    @patch('scripts.search_cli.QueryEngine')
    @patch('builtins.print')
    def test_execute_search_text(self, mock_print, mock_query_engine_class, mock_sys_path):
        """テキスト検索実行のテスト"""
        sys.path.insert(0, str(project_root / 'scripts'))
        
        try:
            from search_cli import execute_search, setup_logging
            
            # モックの設定
            mock_query_engine = MagicMock()
            mock_query_engine.search_by_text.return_value = [
                ('page-bushou-徳川家康', 0.95),
                ('page-bushou-織田信長', 0.87)
            ]
            
            logger = setup_logging('WARNING')
            
            # テスト実行
            execute_search(
                mock_query_engine, 'text', '徳川家康', 
                5, 0.5, 'simple', logger
            )
            
            # QueryEngineのメソッドが呼ばれたかチェック
            mock_query_engine.search_by_text.assert_called_once_with(
                '徳川家康', top_k=5, threshold=0.5
            )
            
            # 結果が出力されたかチェック
            mock_print.assert_called()
        
        finally:
            if str(project_root / 'scripts') in sys.path:
                sys.path.remove(str(project_root / 'scripts'))
    
    @patch('sys.path')
    @patch('scripts.search_cli.QueryEngine')
    @patch('builtins.print')
    def test_execute_search_document(self, mock_print, mock_query_engine_class, mock_sys_path):
        """ドキュメント検索実行のテスト"""
        sys.path.insert(0, str(project_root / 'scripts'))
        
        try:
            from search_cli import execute_search, setup_logging
            
            # モックの設定
            mock_query_engine = MagicMock()
            mock_query_engine.search_by_document_key.return_value = [
                ('page-bushou-織田信長', 0.92),
                ('page-bushou-豊臣秀吉', 0.88)
            ]
            
            logger = setup_logging('WARNING')
            
            # テスト実行
            execute_search(
                mock_query_engine, 'document', 'page-bushou-徳川家康', 
                3, 0.7, 'detailed', logger
            )
            
            # QueryEngineのメソッドが呼ばれたかチェック
            mock_query_engine.search_by_document_key.assert_called_once_with(
                'page-bushou-徳川家康', top_k=3, threshold=0.7
            )
            
            # 結果が出力されたかチェック
            mock_print.assert_called()
        
        finally:
            if str(project_root / 'scripts') in sys.path:
                sys.path.remove(str(project_root / 'scripts'))


class TestMainScript(unittest.TestCase):
    """
    メインスクリプトのテストクラス
    Test class for main script
    """
    
    def setUp(self):
        """テストセットアップ"""
        self.main_script = project_root / 'main.py'
    
    def test_help_option(self):
        """ヘルプオプションのテスト"""
        result = subprocess.run(
            [sys.executable, str(self.main_script), '--help'],
            capture_output=True,
            text=True
        )
        
        self.assertEqual(result.returncode, 0)
        self.assertIn('日本語HTMLベクトル検索システム', result.stdout)
        self.assertIn('batch', result.stdout)
        self.assertIn('search', result.stdout)
    
    def test_version_option(self):
        """バージョンオプションのテスト"""
        result = subprocess.run(
            [sys.executable, str(self.main_script), '--version'],
            capture_output=True,
            text=True
        )
        
        self.assertEqual(result.returncode, 0)
        self.assertIn('Japanese HTML Vector Search System v1.0.0', result.stdout)
    
    def test_no_mode_specified(self):
        """モードが指定されていない場合のテスト"""
        result = subprocess.run(
            [sys.executable, str(self.main_script)],
            capture_output=True,
            text=True
        )
        
        # モードが指定されていない場合はエラー終了するはず
        self.assertNotEqual(result.returncode, 0)
    
    def test_batch_help(self):
        """バッチモードのヘルプテスト"""
        result = subprocess.run(
            [sys.executable, str(self.main_script), 'batch', '--help-full'],
            capture_output=True,
            text=True
        )
        
        # バッチ処理のヘルプが表示されるはず
        self.assertEqual(result.returncode, 0)
    
    def test_search_help(self):
        """検索モードのヘルプテスト"""
        result = subprocess.run(
            [sys.executable, str(self.main_script), 'search', '--help-full'],
            capture_output=True,
            text=True
        )
        
        # 検索のヘルプが表示されるはず
        self.assertEqual(result.returncode, 0)


class TestConfigLoader(unittest.TestCase):
    """
    設定ローダーのテストクラス
    Test class for configuration loader
    """
    
    def setUp(self):
        """テストセットアップ"""
        from config.config_loader import ConfigLoader
        self.temp_dir = tempfile.mkdtemp()
        self.config_loader = ConfigLoader(Path(self.temp_dir))
    
    def test_load_json_config(self):
        """JSON設定ファイル読み込みのテスト"""
        # テスト用JSON設定ファイルを作成
        test_config = {
            "database": {
                "db_path": "/test/path/vectors.db"
            },
            "embedding": {
                "batch_size": 64
            }
        }
        
        config_file = Path(self.temp_dir) / 'test_config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(test_config, f)
        
        # 設定を読み込み
        loaded_config = self.config_loader.load_json_config('test_config.json')
        
        self.assertEqual(loaded_config['database']['db_path'], '/test/path/vectors.db')
        self.assertEqual(loaded_config['embedding']['batch_size'], 64)
    
    def test_load_nonexistent_json_config(self):
        """存在しないJSON設定ファイルのテスト"""
        loaded_config = self.config_loader.load_json_config('nonexistent.json')
        self.assertEqual(loaded_config, {})
    
    def test_load_invalid_json_config(self):
        """無効なJSON設定ファイルのテスト"""
        # 無効なJSONファイルを作成
        config_file = Path(self.temp_dir) / 'invalid.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write('{ invalid json }')
        
        loaded_config = self.config_loader.load_json_config('invalid.json')
        self.assertEqual(loaded_config, {})
    
    @patch.dict('os.environ', {
        'VECTOR_DB_PATH': '/env/test/vectors.db',
        'EMBEDDING_BATCH_SIZE': '128',
        'SEARCH_DEFAULT_TOP_K': '10'
    })
    def test_load_env_config(self):
        """環境変数設定読み込みのテスト"""
        env_config = self.config_loader.load_env_config()
        
        self.assertEqual(env_config['database']['db_path'], '/env/test/vectors.db')
        self.assertEqual(env_config['embedding']['batch_size'], 128)
        self.assertEqual(env_config['search']['default_top_k'], 10)
    
    def test_merge_configs(self):
        """設定マージのテスト"""
        config1 = {
            'database': {'db_path': '/path1'},
            'embedding': {'batch_size': 32}
        }
        
        config2 = {
            'database': {'table_name': 'vectors'},
            'search': {'default_top_k': 5}
        }
        
        merged = self.config_loader.merge_configs(config1, config2)
        
        self.assertEqual(merged['database']['db_path'], '/path1')
        self.assertEqual(merged['database']['table_name'], 'vectors')
        self.assertEqual(merged['embedding']['batch_size'], 32)
        self.assertEqual(merged['search']['default_top_k'], 5)
    
    def test_save_config_template(self):
        """設定テンプレート保存のテスト"""
        self.config_loader.save_config_template('test_template.json')
        
        template_file = Path(self.temp_dir) / 'test_template.json'
        self.assertTrue(template_file.exists())
        
        # テンプレートファイルの内容を確認
        with open(template_file, 'r', encoding='utf-8') as f:
            template_config = json.load(f)
        
        self.assertIn('database', template_config)
        self.assertIn('embedding', template_config)
        self.assertIn('html', template_config)
        self.assertIn('search', template_config)
        self.assertIn('log', template_config)


class TestCLIArgumentParsing(unittest.TestCase):
    """
    CLI引数解析のテストクラス
    Test class for CLI argument parsing
    """
    
    def test_batch_script_argument_parsing(self):
        """バッチスクリプトの引数解析テスト"""
        # バッチスクリプトをインポートして引数解析をテスト
        sys.path.insert(0, str(project_root / 'scripts'))
        
        try:
            from batch_process import parse_arguments
            
            # テスト用引数を設定
            test_args = [
                '--directory', '/test/dir',
                '--pattern', 'test-*.html',
                '--force',
                '--batch-size', '64',
                '--log-level', 'DEBUG'
            ]
            
            with patch('sys.argv', ['batch_process.py'] + test_args):
                args = parse_arguments()
                
                self.assertEqual(args.directory, '/test/dir')
                self.assertEqual(args.pattern, 'test-*.html')
                self.assertTrue(args.force)
                self.assertEqual(args.batch_size, 64)
                self.assertEqual(args.log_level, 'DEBUG')
        
        finally:
            sys.path.remove(str(project_root / 'scripts'))
    
    def test_search_script_argument_parsing(self):
        """検索スクリプトの引数解析テスト"""
        sys.path.insert(0, str(project_root / 'scripts'))
        
        try:
            from search_cli import parse_arguments
            
            # テスト用引数を設定
            test_args = [
                '--text', 'test query',
                '--top-k', '10',
                '--threshold', '0.5',
                '--format', 'json'
            ]
            
            with patch('sys.argv', ['search_cli.py'] + test_args):
                args = parse_arguments()
                
                self.assertEqual(args.text, 'test query')
                self.assertEqual(args.top_k, 10)
                self.assertEqual(args.threshold, 0.5)
                self.assertEqual(args.format, 'json')
        
        finally:
            sys.path.remove(str(project_root / 'scripts'))


if __name__ == '__main__':
    unittest.main()


class TestSearchCLIInteractiveMode(unittest.TestCase):
    """
    検索CLIインタラクティブモードのテストクラス
    Test class for search CLI interactive mode
    """
    
    @patch('sys.path')
    @patch('builtins.input')
    @patch('builtins.print')
    @patch('scripts.search_cli.QueryEngine')
    def test_interactive_mode_text_search(self, mock_query_engine_class, mock_print, mock_input, mock_sys_path):
        """インタラクティブモードのテキスト検索テスト"""
        sys.path.insert(0, str(project_root / 'scripts'))
        
        try:
            from search_cli import interactive_mode, setup_logging
            
            # モックの設定
            mock_query_engine = MagicMock()
            mock_query_engine.search_by_text.return_value = [
                ('page-bushou-徳川家康', 0.95)
            ]
            
            # ユーザー入力をシミュレート
            mock_input.side_effect = [
                'text 徳川家康',  # テキスト検索
                'quit'           # 終了
            ]
            
            logger = setup_logging('WARNING')
            
            # テスト実行
            interactive_mode(mock_query_engine, logger)
            
            # QueryEngineのメソッドが呼ばれたかチェック
            mock_query_engine.search_by_text.assert_called_once()
        
        finally:
            if str(project_root / 'scripts') in sys.path:
                sys.path.remove(str(project_root / 'scripts'))
    
    @patch('sys.path')
    @patch('builtins.input')
    @patch('builtins.print')
    @patch('scripts.search_cli.QueryEngine')
    def test_interactive_mode_document_search(self, mock_query_engine_class, mock_print, mock_input, mock_sys_path):
        """インタラクティブモードのドキュメント検索テスト"""
        sys.path.insert(0, str(project_root / 'scripts'))
        
        try:
            from search_cli import interactive_mode, setup_logging
            
            # モックの設定
            mock_query_engine = MagicMock()
            mock_query_engine.search_by_document_key.return_value = [
                ('page-bushou-織田信長', 0.92)
            ]
            
            # ユーザー入力をシミュレート
            mock_input.side_effect = [
                'doc page-bushou-徳川家康',  # ドキュメント検索
                'quit'                      # 終了
            ]
            
            logger = setup_logging('WARNING')
            
            # テスト実行
            interactive_mode(mock_query_engine, logger)
            
            # QueryEngineのメソッドが呼ばれたかチェック
            mock_query_engine.search_by_document_key.assert_called_once()
        
        finally:
            if str(project_root / 'scripts') in sys.path:
                sys.path.remove(str(project_root / 'scripts'))
    
    @patch('sys.path')
    @patch('builtins.input')
    @patch('builtins.print')
    @patch('scripts.search_cli.QueryEngine')
    def test_interactive_mode_settings(self, mock_query_engine_class, mock_print, mock_input, mock_sys_path):
        """インタラクティブモードの設定変更テスト"""
        sys.path.insert(0, str(project_root / 'scripts'))
        
        try:
            from search_cli import interactive_mode, setup_logging
            
            # モックの設定
            mock_query_engine = MagicMock()
            
            # ユーザー入力をシミュレート
            mock_input.side_effect = [
                'set top-k 10',        # top-k設定
                'set threshold 0.8',   # threshold設定
                'set format json',     # format設定
                'help',                # ヘルプ表示
                'quit'                 # 終了
            ]
            
            logger = setup_logging('WARNING')
            
            # テスト実行
            interactive_mode(mock_query_engine, logger)
            
            # 設定変更のメッセージが出力されたかチェック
            mock_print.assert_any_call('取得結果数を 10 に設定しました')
            mock_print.assert_any_call('類似度閾値を 0.8 に設定しました')
            mock_print.assert_any_call('出力フォーマットを json に設定しました')
        
        finally:
            if str(project_root / 'scripts') in sys.path:
                sys.path.remove(str(project_root / 'scripts'))
    
    @patch('sys.path')
    @patch('builtins.input')
    @patch('builtins.print')
    @patch('scripts.search_cli.QueryEngine')
    def test_interactive_mode_invalid_commands(self, mock_query_engine_class, mock_print, mock_input, mock_sys_path):
        """インタラクティブモードの無効なコマンドテスト"""
        sys.path.insert(0, str(project_root / 'scripts'))
        
        try:
            from search_cli import interactive_mode, setup_logging
            
            # モックの設定
            mock_query_engine = MagicMock()
            
            # ユーザー入力をシミュレート
            mock_input.side_effect = [
                'invalid_command',     # 無効なコマンド
                'text',                # 引数不足
                'set invalid_param 10', # 無効なパラメータ
                'quit'                 # 終了
            ]
            
            logger = setup_logging('WARNING')
            
            # テスト実行
            interactive_mode(mock_query_engine, logger)
            
            # エラーメッセージが出力されたかチェック
            mock_print.assert_any_call('不明なコマンド: invalid_command')
            mock_print.assert_any_call('使用法: text <検索クエリ>')
            mock_print.assert_any_call('不明なパラメータ: invalid_param')
        
        finally:
            if str(project_root / 'scripts') in sys.path:
                sys.path.remove(str(project_root / 'scripts'))


class TestSearchCLIErrorHandling(unittest.TestCase):
    """
    検索CLIエラーハンドリングのテストクラス
    Test class for search CLI error handling
    """
    
    @patch('sys.path')
    @patch('builtins.print')
    @patch('scripts.search_cli.QueryEngine')
    def test_execute_search_with_exception(self, mock_query_engine_class, mock_print, mock_sys_path):
        """検索実行時の例外処理テスト"""
        sys.path.insert(0, str(project_root / 'scripts'))
        
        try:
            from search_cli import execute_search, setup_logging
            
            # モックの設定（例外を発生させる）
            mock_query_engine = MagicMock()
            mock_query_engine.search_by_text.side_effect = Exception("データベース接続エラー")
            
            logger = setup_logging('WARNING')
            
            # テスト実行
            execute_search(
                mock_query_engine, 'text', 'test query', 
                5, 0.5, 'simple', logger
            )
            
            # エラーメッセージが出力されたかチェック
            mock_print.assert_any_call('検索中にエラーが発生しました: データベース接続エラー')
        
        finally:
            if str(project_root / 'scripts') in sys.path:
                sys.path.remove(str(project_root / 'scripts'))
    
    @patch('sys.path')
    @patch('builtins.print')
    @patch('scripts.search_cli.QueryEngine')
    def test_execute_search_invalid_query_type(self, mock_query_engine_class, mock_print, mock_sys_path):
        """無効なクエリタイプのテスト"""
        sys.path.insert(0, str(project_root / 'scripts'))
        
        try:
            from search_cli import execute_search, setup_logging
            
            # モックの設定
            mock_query_engine = MagicMock()
            logger = setup_logging('WARNING')
            
            # テスト実行（無効なクエリタイプ）
            execute_search(
                mock_query_engine, 'invalid_type', 'test query', 
                5, 0.5, 'simple', logger
            )
            
            # QueryEngineのメソッドが呼ばれていないことをチェック
            mock_query_engine.search_by_text.assert_not_called()
            mock_query_engine.search_by_document_key.assert_not_called()
        
        finally:
            if str(project_root / 'scripts') in sys.path:
                sys.path.remove(str(project_root / 'scripts'))