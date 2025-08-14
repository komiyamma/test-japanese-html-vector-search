"""
設定ファイル読み込みユーティリティ
Configuration file loader utility
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from .settings import (
    DATABASE_CONFIG, EMBEDDING_CONFIG, HTML_CONFIG, 
    SEARCH_CONFIG, LOG_CONFIG, PROJECT_ROOT
)


class ConfigLoader:
    """
    設定ファイル読み込みクラス
    Configuration file loader class
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        初期化
        Initialize
        
        Args:
            config_dir: 設定ファイルディレクトリ（デフォルト: PROJECT_ROOT/config）
        """
        self.config_dir = config_dir or PROJECT_ROOT / 'config'
        self.logger = logging.getLogger(__name__)
    
    def load_json_config(self, filename: str) -> Dict[str, Any]:
        """
        JSONファイルから設定を読み込み
        Load configuration from JSON file
        
        Args:
            filename: JSONファイル名
        
        Returns:
            読み込まれた設定辞書
        """
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            self.logger.warning(f"設定ファイルが見つかりません: {config_path}")
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.logger.info(f"設定ファイルを読み込みました: {config_path}")
            return config
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON設定ファイルの解析エラー: {config_path}, {e}")
            return {}
        except Exception as e:
            self.logger.error(f"設定ファイル読み込みエラー: {config_path}, {e}")
            return {}
    
    def load_env_config(self) -> Dict[str, Any]:
        """
        環境変数から設定を読み込み
        Load configuration from environment variables
        
        Returns:
            環境変数から読み込まれた設定辞書
        """
        config = {}
        
        # データベース設定
        if os.getenv('VECTOR_DB_PATH'):
            config['database'] = config.get('database', {})
            config['database']['db_path'] = os.getenv('VECTOR_DB_PATH')
        
        # 埋め込みモデル設定
        if os.getenv('EMBEDDING_MODEL_NAME'):
            config['embedding'] = config.get('embedding', {})
            config['embedding']['model_name'] = os.getenv('EMBEDDING_MODEL_NAME')
        
        if os.getenv('EMBEDDING_BATCH_SIZE'):
            try:
                batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE'))
                config['embedding'] = config.get('embedding', {})
                config['embedding']['batch_size'] = batch_size
            except ValueError:
                self.logger.warning("EMBEDDING_BATCH_SIZE環境変数が無効な値です")
        
        # HTML処理設定
        if os.getenv('HTML_FILE_PATTERN'):
            config['html'] = config.get('html', {})
            config['html']['file_pattern'] = os.getenv('HTML_FILE_PATTERN')
        
        if os.getenv('HTML_MIN_CONTENT_LENGTH'):
            try:
                min_length = int(os.getenv('HTML_MIN_CONTENT_LENGTH'))
                config['html'] = config.get('html', {})
                config['html']['min_content_length'] = min_length
            except ValueError:
                self.logger.warning("HTML_MIN_CONTENT_LENGTH環境変数が無効な値です")
        
        # 検索設定
        if os.getenv('SEARCH_DEFAULT_TOP_K'):
            try:
                top_k = int(os.getenv('SEARCH_DEFAULT_TOP_K'))
                config['search'] = config.get('search', {})
                config['search']['default_top_k'] = top_k
            except ValueError:
                self.logger.warning("SEARCH_DEFAULT_TOP_K環境変数が無効な値です")
        
        if os.getenv('SEARCH_SIMILARITY_THRESHOLD'):
            try:
                threshold = float(os.getenv('SEARCH_SIMILARITY_THRESHOLD'))
                config['search'] = config.get('search', {})
                config['search']['similarity_threshold'] = threshold
            except ValueError:
                self.logger.warning("SEARCH_SIMILARITY_THRESHOLD環境変数が無効な値です")
        
        # ログ設定
        if os.getenv('LOG_LEVEL'):
            config['log'] = config.get('log', {})
            config['log']['level'] = os.getenv('LOG_LEVEL')
        
        if os.getenv('LOG_FILE'):
            config['log'] = config.get('log', {})
            config['log']['log_file'] = os.getenv('LOG_FILE')
        
        if config:
            self.logger.info("環境変数から設定を読み込みました")
        
        return config
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        複数の設定辞書をマージ
        Merge multiple configuration dictionaries
        
        Args:
            *configs: マージする設定辞書のリスト
        
        Returns:
            マージされた設定辞書
        """
        merged = {}
        
        for config in configs:
            for key, value in config.items():
                if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                    merged[key].update(value)
                else:
                    merged[key] = value
        
        return merged
    
    def get_merged_config(self, custom_config_file: Optional[str] = None) -> Dict[str, Any]:
        """
        すべての設定ソースからマージされた設定を取得
        Get merged configuration from all sources
        
        Args:
            custom_config_file: カスタム設定ファイル名
        
        Returns:
            マージされた設定辞書
        """
        # デフォルト設定
        default_config = {
            'database': DATABASE_CONFIG,
            'embedding': EMBEDDING_CONFIG,
            'html': HTML_CONFIG,
            'search': SEARCH_CONFIG,
            'log': LOG_CONFIG
        }
        
        # JSONファイルから設定を読み込み
        json_config = {}
        if custom_config_file:
            json_config = self.load_json_config(custom_config_file)
        else:
            # デフォルトの設定ファイルを試行
            for filename in ['config.json', 'settings.json']:
                json_config = self.load_json_config(filename)
                if json_config:
                    break
        
        # 環境変数から設定を読み込み
        env_config = self.load_env_config()
        
        # 設定をマージ（優先度: 環境変数 > JSONファイル > デフォルト）
        merged_config = self.merge_configs(default_config, json_config, env_config)
        
        return merged_config
    
    def save_config_template(self, filename: str = 'config.template.json') -> None:
        """
        設定テンプレートファイルを保存
        Save configuration template file
        
        Args:
            filename: テンプレートファイル名
        """
        template_config = {
            "database": {
                "db_path": str(DATABASE_CONFIG['db_path']),
                "table_name": DATABASE_CONFIG['table_name']
            },
            "embedding": {
                "model_name": EMBEDDING_CONFIG['model_name'],
                "batch_size": EMBEDDING_CONFIG['batch_size'],
                "max_seq_length": EMBEDDING_CONFIG['max_seq_length']
            },
            "html": {
                "file_pattern": HTML_CONFIG['file_pattern'],
                "encoding": HTML_CONFIG['encoding'],
                "min_content_length": HTML_CONFIG['min_content_length'],
                "max_content_length": HTML_CONFIG['max_content_length']
            },
            "search": {
                "default_top_k": SEARCH_CONFIG['default_top_k'],
                "similarity_threshold": SEARCH_CONFIG['similarity_threshold']
            },
            "log": {
                "level": LOG_CONFIG['level'],
                "format": LOG_CONFIG['format'],
                "log_file": str(LOG_CONFIG['log_file'])
            }
        }
        
        template_path = self.config_dir / filename
        
        try:
            with open(template_path, 'w', encoding='utf-8') as f:
                json.dump(template_config, f, ensure_ascii=False, indent=2)
            self.logger.info(f"設定テンプレートを保存しました: {template_path}")
        except Exception as e:
            self.logger.error(f"設定テンプレート保存エラー: {template_path}, {e}")


def load_config(custom_config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    設定を読み込む便利関数
    Convenience function to load configuration
    
    Args:
        custom_config_file: カスタム設定ファイル名
    
    Returns:
        マージされた設定辞書
    """
    loader = ConfigLoader()
    return loader.get_merged_config(custom_config_file)


def create_config_template() -> None:
    """
    設定テンプレートを作成する便利関数
    Convenience function to create configuration template
    """
    loader = ConfigLoader()
    loader.save_config_template()