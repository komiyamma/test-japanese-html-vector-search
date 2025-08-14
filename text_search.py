#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
テキストクエリでベクトル検索を行うスクリプト
"""

import sqlite3
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def cosine_similarity(vec1, vec2):
    """コサイン類似度を計算"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def search_by_text(db_path, query_text, top_k=5):
    """テキストクエリでベクトル検索"""
    
    print(f"検索クエリ: '{query_text}'")
    print("ベクトル化モデルを読み込み中...")
    
    # ベクトル化モデルを初期化
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # クエリテキストをベクトル化
    query_vector = model.encode(query_text)
    
    # データベース接続
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 全ベクトルデータを取得
    cursor.execute("SELECT document_key, vector_data FROM document_vectors")
    all_data = cursor.fetchall()
    
    if not all_data:
        print("データベースにデータが見つかりません")
        return
    
    print(f"データベース内のドキュメント数: {len(all_data)}")
    print("類似度を計算中...")
    
    # 類似度を計算
    similarities = []
    for key, vector_blob in all_data:
        vector = pickle.loads(vector_blob)
        similarity = cosine_similarity(query_vector, vector)
        similarities.append((key, similarity))
    
    # 類似度でソート（降順）
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 結果を表示
    print(f"\n=== 検索結果（上位{min(top_k, len(similarities))}件）===")
    
    for i, (key, score) in enumerate(similarities[:top_k], 1):
        print(f"{i}. {key}")
        print(f"   スコア: {score:.4f}")
        print()
    
    conn.close()

def main():
    """メイン処理"""
    
    # 設定
    db_path = r"G:\repogitory\deep-sengoku.net\vectors.db"
    
    # 検索クエリ（ここを変更してください）
    query_text = "織田信長"  # 20文字程度の例
    
    print(f"データベース: {db_path}")
    print()
    
    # データベースファイルの存在確認
    if not Path(db_path).exists():
        print(f"エラー: データベースファイルが見つかりません: {db_path}")
        return
    
    # テキスト検索実行
    search_by_text(db_path, query_text, top_k=5)

if __name__ == "__main__":
    main()