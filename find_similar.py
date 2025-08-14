#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指定したキーに類似度が高い上位20個のドキュメントを表示するスクリプト
"""

import sqlite3
import pickle
import numpy as np
from pathlib import Path

def cosine_similarity(vec1, vec2):
    """コサイン類似度を計算"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def find_similar_documents(db_path, target_key, top_k=20):
    """指定したキーに類似したドキュメントを検索"""
    
    # データベース接続
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 全ベクトルデータを取得
    cursor.execute("SELECT document_key, vector_data FROM document_vectors")
    all_data = cursor.fetchall()
    
    if not all_data:
        print("データベースにデータが見つかりません")
        return
    
    # ターゲットベクトルを取得
    target_vector = None
    for key, vector_blob in all_data:
        if key == target_key:
            target_vector = pickle.loads(vector_blob)
            break
    
    if target_vector is None:
        print(f"キー '{target_key}' が見つかりません")
        print("利用可能なキー:")
        for key, _ in all_data[:10]:  # 最初の10個を表示
            print(f"  - {key}")
        return
    
    # 類似度を計算
    similarities = []
    for key, vector_blob in all_data:
        if key != target_key:  # 自分自身は除外
            vector = pickle.loads(vector_blob)
            similarity = cosine_similarity(target_vector, vector)
            similarities.append((key, similarity))
    
    # 類似度でソート（降順）
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 結果を表示
    print(f"=== '{target_key}' に類似したドキュメント（上位{min(top_k, len(similarities))}件）===")
    print()
    
    for i, (key, score) in enumerate(similarities[:top_k], 1):
        print(f"{i:2d}. {key}")
        print(f"    類似度スコア: {score:.4f}")
        print()
    
    conn.close()

def main():
    """メイン処理"""
    
    # 設定
    db_path = r"G:\repogitory\deep-sengoku.net\vectors.db"
    target_key = "page-bushou-織田信長"  # 検索対象のキー
    
    print(f"データベース: {db_path}")
    print(f"検索対象: {target_key}")
    print()
    
    # データベースファイルの存在確認
    if not Path(db_path).exists():
        print(f"エラー: データベースファイルが見つかりません: {db_path}")
        return
    
    # 類似ドキュメント検索
    find_similar_documents(db_path, target_key, top_k=20)

if __name__ == "__main__":
    main()