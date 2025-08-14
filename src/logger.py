"""
ログ設定モジュール
Logging configuration module
"""

import logging
import os
from pathlib import Path
from config.settings import LOG_CONFIG


def setup_logger(name: str = __name__) -> logging.Logger:
    """
    ログ設定を初期化し、ロガーを返す
    Initialize logging configuration and return logger
    
    Args:
        name (str): ロガー名
        
    Returns:
        logging.Logger: 設定済みロガー
    """
    # ログディレクトリを作成
    log_dir = LOG_CONFIG['log_file'].parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # ロガーを取得
    logger = logging.getLogger(name)
    
    # 既に設定済みの場合はそのまま返す
    if logger.handlers:
        return logger
    
    # ログレベルを設定
    logger.setLevel(getattr(logging, LOG_CONFIG['level']))
    
    # フォーマッターを作成
    formatter = logging.Formatter(LOG_CONFIG['format'])
    
    # ファイルハンドラーを作成
    file_handler = logging.FileHandler(
        LOG_CONFIG['log_file'], 
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, LOG_CONFIG['level']))
    file_handler.setFormatter(formatter)
    
    # コンソールハンドラーを作成
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOG_CONFIG['level']))
    console_handler.setFormatter(formatter)
    
    # ハンドラーをロガーに追加
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# デフォルトロガーを作成
default_logger = setup_logger('japanese_vector_search')