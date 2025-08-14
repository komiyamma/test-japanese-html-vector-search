#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G:/htmlフォルダのHTMLファイルを最低限の機能でベクトル化するシンプルスクリプト
"""

import sys
import sqlite3
import pickle
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

def extract_text_from_html(html_file):
    """HTMLファイルからテキストを抽出"""
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            return soup.get_text(strip=True)
    except:
        return ""

def create_database(db_path):
    """データベースを作成"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_key TEXT UNIQUE NOT NULL,
            vector_data BLOB NOT NULL,
            text_content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    return conn

def main():
    """G:/htmlフォルダのHTMLファイルを処理"""
    
    # 設定
    html_dir = Path(r"G:\repogitory\deep-sengoku.net")
    db_file = r"G:\repogitory\deep-sengoku.net\vectors.db"
    
    print(f"処理開始: {html_dir}")
    print(f"データベース: {db_file}")
    
    # ディレクトリ存在チェック
    if not html_dir.exists():
        print(f"エラー: {html_dir} が見つかりません")
        return
    
    # ベクトル化モデルを初期化
    print("ベクトル化モデルを読み込み中...")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # データベース作成
    conn = create_database(db_file)
    cursor = conn.cursor()
    
    # page-*-*.htmlファイルを検索
    html_files = list(html_dir.glob("page-*-*.html"))
    print(f"対象ファイル数: {len(html_files)}")
    
    success_count = 0
    error_count = 0
    
    for html_file in html_files:
        try:
            # ファイル名からキーを生成（拡張子を除く）
            doc_key = html_file.stem
            
            # テキスト抽出
            text_content = extract_text_from_html(html_file)
            
            if text_content:
                # ベクトル化
                vector = model.encode(text_content)
                vector_blob = pickle.dumps(vector)
                
                # データベースに保存
                cursor.execute('''
                    INSERT OR REPLACE INTO document_vectors 
                    (document_key, vector_data, text_content) VALUES (?, ?, ?)
                ''', (doc_key, vector_blob, text_content))
                
                success_count += 1
                print(f"処理完了: {doc_key}")
            else:
                print(f"テキスト抽出失敗: {html_file.name}")
                error_count += 1
                
        except Exception as e:
            print(f"エラー: {html_file.name} - {e}")
            error_count += 1
    
    # データベースを閉じる
    conn.commit()
    conn.close()
    
    print(f"完了: 成功 {success_count}件, エラー {error_count}件")
    print(f"データベースファイル: {Path(db_file).absolute()}")

if __name__ == "__main__":
    main()