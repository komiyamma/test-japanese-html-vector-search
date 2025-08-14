# トラブルシューティングガイド

このドキュメントでは、日本語HTMLベクトル検索システムでよく発生する問題とその解決方法について説明します。

## インストール関連の問題

### 1. sentence-transformersのインストールエラー

**症状:**
```
ERROR: Could not find a version that satisfies the requirement sentence-transformers
```

**原因:**
- Python バージョンが古い
- pip が古い
- ネットワーク接続の問題

**解決方法:**

```bash
# Python バージョンの確認（3.8以上が必要）
python --version

# pip のアップグレード
python -m pip install --upgrade pip

# sentence-transformers の個別インストール
pip install sentence-transformers

# torch が先にインストールされていない場合
pip install torch torchvision torchaudio

# 全依存関係の再インストール
pip install -r requirements.txt --force-reinstall
```

### 2. PyTorchのインストールエラー

**症状:**
```
ERROR: Could not install packages due to an EnvironmentError: [Errno 28] No space left on device
```

**原因:**
- ディスク容量不足
- 一時ディレクトリの容量不足

**解決方法:**

```bash
# ディスク容量の確認
df -h

# pip キャッシュのクリア
pip cache purge

# 一時ディレクトリを指定してインストール
pip install --cache-dir /tmp/pip-cache sentence-transformers

# CPU版のPyTorchを使用（GPUが不要な場合）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. 依存関係の競合

**症状:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
```

**解決方法:**

```bash
# 仮想環境の作成（推奨）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate  # Windows

# クリーンインストール
pip install -r requirements.txt

# 依存関係の確認
pip check
```

## 実行時エラー

### 4. データベース関連エラー

**症状:**
```
sqlite3.OperationalError: database is locked
```

**原因:**
- 複数のプロセスが同時にデータベースにアクセス
- プロセスが異常終了してロックが残っている

**解決方法:**

```bash
# データベースファイルの確認
ls -la data/vectors.db*

# ロックファイルの削除
rm -f data/vectors.db-wal data/vectors.db-shm

# データベースの整合性チェック
sqlite3 data/vectors.db "PRAGMA integrity_check;"

# 新しいデータベースファイルで再作成
mv data/vectors.db data/vectors.db.backup
python scripts/batch_process.py --force
```

### 5. メモリ不足エラー

**症状:**
```
RuntimeError: [enforce fail at alloc_cpu.cpp:75] data. DefaultCPUAllocator: not enough memory
```

**原因:**
- バッチサイズが大きすぎる
- 大きなHTMLファイルの処理
- システムメモリ不足

**解決方法:**

```bash
# バッチサイズを小さくして実行
python scripts/batch_process.py --batch-size 8

# メモリ使用量の監視
python scripts/batch_process.py --log-level DEBUG

# 環境変数でバッチサイズを設定
export EMBEDDING_BATCH_SIZE=16
python scripts/batch_process.py
```

**コード内での対処:**

```python
# メモリ効率的な処理
import gc
import torch

def process_with_memory_management():
    try:
        # 処理実行
        result = embedder.embed_batch(texts)
        
        # メモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return result
    except RuntimeError as e:
        if "memory" in str(e).lower():
            # バッチサイズを半分にして再試行
            smaller_batch = len(texts) // 2
            return process_smaller_batch(texts, smaller_batch)
        raise
```

### 6. 日本語エンコーディングエラー

**症状:**
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x82 in position 1: invalid start byte
```

**原因:**
- HTMLファイルが UTF-8 以外でエンコードされている
- ファイルの文字コード検出に失敗

**解決方法:**

```python
# HTMLProcessor での対処例
import chardet
from pathlib import Path

def extract_text_with_encoding_detection(self, html_file_path: str) -> str:
    """エンコーディング自動検出付きテキスト抽出"""
    
    try:
        # まず UTF-8 で試行
        with open(html_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # エンコーディング自動検出
        with open(html_file_path, 'rb') as f:
            raw_data = f.read()
            detected = chardet.detect(raw_data)
            encoding = detected['encoding']
            
        # 検出されたエンコーディングで再試行
        with open(html_file_path, 'r', encoding=encoding) as f:
            content = f.read()
    
    # BeautifulSoup でパース
    soup = BeautifulSoup(content, 'html.parser')
    return soup.get_text()
```

**手動での文字コード変換:**

```bash
# ファイルの文字コード確認
file -i page-bushou-*.html

# Shift_JIS から UTF-8 への変換
iconv -f SHIFT_JIS -t UTF-8 input.html > output.html

# 一括変換
for file in page-bushou-*.html; do
    iconv -f SHIFT_JIS -t UTF-8 "$file" > "utf8_$file"
done
```

### 7. モデル読み込みエラー

**症状:**
```
OSError: Can't load tokenizer for 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
```

**原因:**
- ネットワーク接続の問題
- Hugging Face Hub へのアクセス制限
- モデルキャッシュの破損

**解決方法:**

```bash
# モデルキャッシュの場所確認
python -c "from transformers import TRANSFORMERS_CACHE; print(TRANSFORMERS_CACHE)"

# キャッシュのクリア
rm -rf ~/.cache/huggingface/transformers/

# 手動でモデルをダウンロード
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
print('モデルのダウンロード完了')
"

# プロキシ環境での設定
export HF_HUB_OFFLINE=1  # オフラインモード
export TRANSFORMERS_OFFLINE=1
```

## 検索関連の問題

### 8. 検索結果が返されない

**症状:**
- 検索クエリに対して結果が0件
- 明らかに関連するドキュメントがあるのに見つからない

**原因:**
- 類似度閾値が高すぎる
- ベクトルデータが正しく保存されていない
- クエリテキストの前処理に問題

**診断方法:**

```python
# データベース内容の確認
def diagnose_database():
    db_manager = DatabaseManager("data/vectors.db")
    all_vectors = db_manager.get_all_vectors()
    
    print(f"保存されているドキュメント数: {len(all_vectors)}")
    
    for key, vector in list(all_vectors.items())[:5]:
        print(f"キー: {key}, ベクトル形状: {vector.shape}")
        print(f"ベクトルの統計: min={vector.min():.4f}, max={vector.max():.4f}, mean={vector.mean():.4f}")

# 類似度の詳細確認
def diagnose_similarity(query_text: str):
    embedder = VectorEmbedder()
    db_manager = DatabaseManager("data/vectors.db")
    calculator = SimilarityCalculator()
    
    query_vector = embedder.embed_text(query_text)
    all_vectors = db_manager.get_all_vectors()
    
    similarities = []
    for key, vector in all_vectors.items():
        similarity = calculator.cosine_similarity(query_vector, vector)
        similarities.append((key, similarity))
    
    # 類似度順でソート
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"クエリ: '{query_text}'")
    print("全ドキュメントとの類似度:")
    for key, sim in similarities[:10]:
        print(f"  {key}: {sim:.4f}")
```

**解決方法:**

```bash
# 閾値を下げて検索
python scripts/search_cli.py --text "徳川家康" --threshold 0.0

# より多くの結果を取得
python scripts/search_cli.py --text "徳川家康" --top-k 20

# デバッグモードで実行
python scripts/search_cli.py --text "徳川家康" --log-level DEBUG
```

### 9. 検索が遅い

**症状:**
- 検索に時間がかかりすぎる
- 大量のドキュメントがある場合の性能問題

**原因:**
- データベースにインデックスがない
- 類似度計算の最適化不足
- メモリ不足による スワップ

**解決方法:**

```sql
-- データベースにインデックスを追加
CREATE INDEX IF NOT EXISTS idx_document_key ON document_vectors(document_key);
CREATE INDEX IF NOT EXISTS idx_created_at ON document_vectors(created_at);
```

```python
# 類似度計算の最適化
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def optimized_similarity_search(query_vector, document_vectors, top_k=5):
    """最適化された類似度検索"""
    
    # NumPy配列に変換
    doc_keys = list(document_vectors.keys())
    doc_matrix = np.array(list(document_vectors.values()))
    
    # scikit-learnの最適化された実装を使用
    similarities = cosine_similarity([query_vector], doc_matrix)[0]
    
    # 上位k件を効率的に取得
    top_indices = np.argpartition(similarities, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
    
    results = [(doc_keys[i], similarities[i]) for i in top_indices]
    return results
```

## ログとデバッグ

### 10. ログが出力されない

**症状:**
- ログファイルが作成されない
- コンソールにログが表示されない

**解決方法:**

```python
# ログ設定の確認
import logging

# 現在のログレベル確認
logger = logging.getLogger()
print(f"現在のログレベル: {logger.level}")
print(f"ハンドラー数: {len(logger.handlers)}")

# ログ設定のリセット
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)
```

```bash
# ログディレクトリの権限確認
ls -la logs/
chmod 755 logs/
touch logs/test.log  # 書き込み権限の確認
```

### 11. デバッグ情報の取得

**詳細なシステム情報の取得:**

```python
#!/usr/bin/env python3
"""
システム診断スクリプト
"""

import sys
import os
import platform
import sqlite3
from pathlib import Path

def system_diagnosis():
    """システム診断情報を出力"""
    
    print("=== システム診断情報 ===")
    print(f"Python バージョン: {sys.version}")
    print(f"プラットフォーム: {platform.platform()}")
    print(f"アーキテクチャ: {platform.architecture()}")
    print(f"現在のディレクトリ: {os.getcwd()}")
    print()
    
    # 依存関係の確認
    print("=== 依存関係の確認 ===")
    required_packages = [
        'sentence_transformers',
        'beautifulsoup4',
        'numpy',
        'sqlite3',
        'torch'
    ]
    
    for package in required_packages:
        try:
            if package == 'sqlite3':
                import sqlite3
                print(f"✓ {package}: {sqlite3.sqlite_version}")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'バージョン不明')
                print(f"✓ {package}: {version}")
        except ImportError:
            print(f"✗ {package}: インストールされていません")
    print()
    
    # ファイル構造の確認
    print("=== ファイル構造の確認 ===")
    important_paths = [
        'src/',
        'config/',
        'scripts/',
        'tests/',
        'data/',
        'logs/',
        'requirements.txt',
        'main.py'
    ]
    
    for path in important_paths:
        path_obj = Path(path)
        if path_obj.exists():
            if path_obj.is_dir():
                file_count = len(list(path_obj.glob('*')))
                print(f"✓ {path}: ディレクトリ ({file_count}ファイル)")
            else:
                size = path_obj.stat().st_size
                print(f"✓ {path}: ファイル ({size}バイト)")
        else:
            print(f"✗ {path}: 存在しません")
    print()
    
    # データベースの確認
    print("=== データベースの確認 ===")
    db_path = Path("data/vectors.db")
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # テーブル一覧
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"テーブル数: {len(tables)}")
            
            # レコード数
            if tables:
                cursor.execute("SELECT COUNT(*) FROM document_vectors;")
                count = cursor.fetchone()[0]
                print(f"保存されているベクトル数: {count}")
            
            conn.close()
            print("✓ データベース: 正常")
        except Exception as e:
            print(f"✗ データベースエラー: {e}")
    else:
        print("✗ データベースファイルが存在しません")
    print()
    
    # HTMLファイルの確認
    print("=== HTMLファイルの確認 ===")
    html_files = list(Path(".").glob("page-*.html"))
    print(f"HTMLファイル数: {len(html_files)}")
    
    if html_files:
        print("サンプルファイル:")
        for file_path in html_files[:5]:
            size = file_path.stat().st_size
            print(f"  {file_path.name}: {size}バイト")
    print()

if __name__ == "__main__":
    system_diagnosis()
```

## パフォーマンス最適化

### 12. 処理速度の改善

**バッチ処理の最適化:**

```python
# 並列処理の実装
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def parallel_batch_processing(html_files, max_workers=None):
    """並列バッチ処理"""
    
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 4)  # CPUコア数と4の小さい方
    
    def process_single_file(html_file):
        try:
            # 各プロセスで独立したインスタンスを作成
            html_processor = HTMLProcessor()
            vector_embedder = VectorEmbedder()
            database_manager = DatabaseManager("data/vectors.db")
            
            # ファイル処理
            text = html_processor.extract_text(html_file)
            vector = vector_embedder.embed_text(text)
            file_key = html_processor.get_file_key(html_file)
            database_manager.store_vector(file_key, vector)
            
            return {"success": True, "file": html_file}
        except Exception as e:
            return {"success": False, "file": html_file, "error": str(e)}
    
    # 並列実行
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_file, html_files))
    
    return results
```

**メモリ使用量の最適化:**

```python
# ジェネレータを使用したメモリ効率的な処理
def memory_efficient_processing(html_files, batch_size=32):
    """メモリ効率的な処理"""
    
    def file_generator():
        for html_file in html_files:
            try:
                text = html_processor.extract_text(html_file)
                yield html_file, text
            except Exception as e:
                logger.error(f"ファイル読み込みエラー {html_file}: {e}")
    
    # バッチ単位で処理
    batch = []
    for html_file, text in file_generator():
        batch.append((html_file, text))
        
        if len(batch) >= batch_size:
            process_batch(batch)
            batch = []  # メモリ解放
    
    # 残りのバッチを処理
    if batch:
        process_batch(batch)
```

このトラブルシューティングガイドを参考に、問題が発生した際の診断と解決を行ってください。問題が解決しない場合は、システム診断スクリプトを実行して詳細な情報を収集し、ログファイルを確認することをお勧めします。