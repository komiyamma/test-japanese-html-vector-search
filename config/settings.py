"""
設定ファイル
Configuration file for Japanese HTML Vector Search System
"""

import os
from pathlib import Path

# プロジェクトルートディレクトリ
PROJECT_ROOT = Path(__file__).parent.parent

# データベース設定
DATABASE_CONFIG = {
    'db_path': PROJECT_ROOT / 'data' / 'vectors.db',
    'table_name': 'document_vectors'
}

# ベクトル埋め込み設定
EMBEDDING_CONFIG = {
    'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'batch_size': 32,
    'max_seq_length': 512
}

# HTMLファイル処理設定
HTML_CONFIG = {
    'file_pattern': 'page-*.html',
    'encoding': 'utf-8',
    'min_content_length': 1000,
    'max_content_length': 30000
}

# 類似度検索設定
SEARCH_CONFIG = {
    'default_top_k': 5,
    'similarity_threshold': 0.1
}

# ログ設定
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': PROJECT_ROOT / 'logs' / 'system.log'
}