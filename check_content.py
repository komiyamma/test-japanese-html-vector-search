#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データベース内のテキスト内容を確認するスクリプト
"""

import sqlite3
from pathlib import Path

def check_document_content(db_path, target_key):
    """指定したキーのテキスト内容を表示"""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 指定したキーのテキストを取得
    cursor.execute("SELECT document_key, text_content FROM document_vectors WHERE document_key = ?", (target_key,))
    result = cursor.fetchone()
    
    if result:
        key, text = result
        print(f"=== {key} のテキスト内容 ===")
        print(f"文字数: {len(text)}")
        print("=" * 50)
        print(text[:1000])  # 最初の1000文字を表示
        if len(text) > 1000:
            print("\n... (省略) ...")
        print("=" * 50)
    else:
        print(f"キー '{target_key}' が見つかりません")
    
    conn.close()

def main():
    db_path = r"G:\repogitory\deep-sengoku.net\vectors.db"
    target_key = "page-bushou-織田信長"
    
    check_document_content(db_path, target_key)

if __name__ == "__main__":
    main()