# APIリファレンス

このドキュメントでは、日本語HTMLベクトル検索システムの各コンポーネントのAPIについて詳しく説明します。

## HTMLProcessor

HTMLファイルからテキストを抽出し、処理するためのクラスです。

### クラス定義

```python
from src.html_processor import HTMLProcessor

processor = HTMLProcessor()
```

### メソッド

#### `extract_text(html_file_path: str) -> str`

HTMLファイルからメインコンテンツのテキストを抽出します。

**パラメータ:**
- `html_file_path` (str): HTMLファイルのパス

**戻り値:**
- `str`: 抽出されたテキストコンテンツ

**例外:**
- `FileNotFoundError`: ファイルが見つからない場合
- `UnicodeDecodeError`: エンコーディングエラーの場合

**使用例:**
```python
processor = HTMLProcessor()
text = processor.extract_text("page-bushou-徳川家康.html")
print(f"抽出されたテキスト: {text[:100]}...")
```

#### `get_file_key(html_file_path: str) -> str`

HTMLファイルパスからデータベースキーを生成します。

**パラメータ:**
- `html_file_path` (str): HTMLファイルのパス

**戻り値:**
- `str`: 拡張子を除いたファイル名

**使用例:**
```python
key = processor.get_file_key("page-bushou-徳川家康.html")
print(key)  # "page-bushou-徳川家康"
```

#### `validate_content_length(text: str) -> bool`

テキストコンテンツの長さを検証します。

**パラメータ:**
- `text` (str): 検証するテキスト

**戻り値:**
- `bool`: コンテンツが有効な長さの場合True

## VectorEmbedder

テキストをベクトル化するためのクラスです。

### クラス定義

```python
from src.vector_embedder import VectorEmbedder

embedder = VectorEmbedder(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
```

### コンストラクタ

#### `__init__(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")`

**パラメータ:**
- `model_name` (str): 使用するSentence Transformersモデル名

### メソッド

#### `embed_text(text: str) -> np.ndarray`

単一のテキストをベクトル化します。

**パラメータ:**
- `text` (str): ベクトル化するテキスト

**戻り値:**
- `np.ndarray`: ベクトル化されたテキスト（384次元）

**使用例:**
```python
embedder = VectorEmbedder()
vector = embedder.embed_text("徳川家康は江戸幕府の初代将軍です。")
print(f"ベクトル次元: {vector.shape}")  # (384,)
```

#### `embed_batch(texts: List[str]) -> List[np.ndarray]`

複数のテキストを一括でベクトル化します。

**パラメータ:**
- `texts` (List[str]): ベクトル化するテキストのリスト

**戻り値:**
- `List[np.ndarray]`: ベクトル化されたテキストのリスト

**使用例:**
```python
texts = ["徳川家康", "織田信長", "豊臣秀吉"]
vectors = embedder.embed_batch(texts)
print(f"処理されたテキスト数: {len(vectors)}")
```

## DatabaseManager

SQLiteデータベースでベクトルデータを管理するクラスです。

### クラス定義

```python
from src.database_manager import DatabaseManager

db_manager = DatabaseManager(db_path="vectors.db")
```

### コンストラクタ

#### `__init__(db_path: str = "vectors.db")`

**パラメータ:**
- `db_path` (str): SQLiteデータベースファイルのパス

### メソッド

#### `create_table() -> None`

ベクトルデータ保存用のテーブルを作成します。

**使用例:**
```python
db_manager = DatabaseManager()
db_manager.create_table()
```

#### `store_vector(key: str, vector: np.ndarray) -> None`

ベクトルデータをデータベースに保存します。

**パラメータ:**
- `key` (str): ドキュメントキー
- `vector` (np.ndarray): ベクトルデータ

**使用例:**
```python
import numpy as np

vector = np.random.rand(384)
db_manager.store_vector("page-bushou-徳川家康", vector)
```

#### `get_vector(key: str) -> Optional[np.ndarray]`

指定されたキーのベクトルデータを取得します。

**パラメータ:**
- `key` (str): ドキュメントキー

**戻り値:**
- `Optional[np.ndarray]`: ベクトルデータ（存在しない場合はNone）

**使用例:**
```python
vector = db_manager.get_vector("page-bushou-徳川家康")
if vector is not None:
    print(f"ベクトル次元: {vector.shape}")
```

#### `get_all_vectors() -> Dict[str, np.ndarray]`

すべてのベクトルデータを取得します。

**戻り値:**
- `Dict[str, np.ndarray]`: キーとベクトルデータの辞書

**使用例:**
```python
all_vectors = db_manager.get_all_vectors()
print(f"保存されているドキュメント数: {len(all_vectors)}")
```

#### `update_vector(key: str, vector: np.ndarray) -> None`

既存のベクトルデータを更新します。

**パラメータ:**
- `key` (str): ドキュメントキー
- `vector` (np.ndarray): 新しいベクトルデータ

## SimilarityCalculator

ベクトル間の類似度を計算するクラスです。

### クラス定義

```python
from src.similarity_calculator import SimilarityCalculator

calculator = SimilarityCalculator()
```

### メソッド

#### `cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float`

2つのベクトル間のコサイン類似度を計算します。

**パラメータ:**
- `vector1` (np.ndarray): 第1のベクトル
- `vector2` (np.ndarray): 第2のベクトル

**戻り値:**
- `float`: コサイン類似度（0.0〜1.0）

**使用例:**
```python
import numpy as np

vector1 = np.random.rand(384)
vector2 = np.random.rand(384)
similarity = calculator.cosine_similarity(vector1, vector2)
print(f"類似度: {similarity:.4f}")
```

#### `find_similar_documents(query_vector: np.ndarray, document_vectors: Dict[str, np.ndarray], top_k: int = 5) -> List[Tuple[str, float]]`

クエリベクトルに類似したドキュメントを検索します。

**パラメータ:**
- `query_vector` (np.ndarray): クエリベクトル
- `document_vectors` (Dict[str, np.ndarray]): ドキュメントベクトルの辞書
- `top_k` (int): 取得する結果数（デフォルト: 5）

**戻り値:**
- `List[Tuple[str, float]]`: (ドキュメントキー, 類似度スコア)のリスト

**使用例:**
```python
query_vector = np.random.rand(384)
document_vectors = {
    "doc1": np.random.rand(384),
    "doc2": np.random.rand(384),
    "doc3": np.random.rand(384)
}

results = calculator.find_similar_documents(query_vector, document_vectors, top_k=2)
for doc_key, score in results:
    print(f"{doc_key}: {score:.4f}")
```

## BatchProcessor

複数のHTMLファイルを一括処理するクラスです。

### クラス定義

```python
from src.batch_processor import BatchProcessor

processor = BatchProcessor(
    html_processor=html_processor,
    vector_embedder=embedder,
    database_manager=db_manager
)
```

### コンストラクタ

#### `__init__(html_processor: HTMLProcessor, vector_embedder: VectorEmbedder, database_manager: DatabaseManager)`

**パラメータ:**
- `html_processor` (HTMLProcessor): HTMLプロセッサインスタンス
- `vector_embedder` (VectorEmbedder): ベクトル埋め込みインスタンス
- `database_manager` (DatabaseManager): データベースマネージャインスタンス

### メソッド

#### `process_directory(directory: str = ".", pattern: str = "page-*.html", force: bool = False) -> Dict[str, Any]`

指定されたディレクトリ内のHTMLファイルを一括処理します。

**パラメータ:**
- `directory` (str): 処理対象ディレクトリ（デフォルト: "."）
- `pattern` (str): ファイルパターン（デフォルト: "page-*.html"）
- `force` (bool): 既存データを強制更新するか（デフォルト: False）

**戻り値:**
- `Dict[str, Any]`: 処理結果の統計情報

**使用例:**
```python
result = processor.process_directory(
    directory="./html_files",
    pattern="page-bushou-*.html",
    force=True
)
print(f"処理されたファイル数: {result['processed_files']}")
print(f"エラー数: {result['errors']}")
```

## QueryEngine

ベクトルデータベースに対してクエリを実行するクラスです。

### クラス定義

```python
from src.query_engine import QueryEngine

query_engine = QueryEngine(
    vector_embedder=embedder,
    database_manager=db_manager,
    similarity_calculator=calculator
)
```

### コンストラクタ

#### `__init__(vector_embedder: VectorEmbedder, database_manager: DatabaseManager, similarity_calculator: SimilarityCalculator)`

**パラメータ:**
- `vector_embedder` (VectorEmbedder): ベクトル埋め込みインスタンス
- `database_manager` (DatabaseManager): データベースマネージャインスタンス
- `similarity_calculator` (SimilarityCalculator): 類似度計算インスタンス

### メソッド

#### `search_by_text(query_text: str, top_k: int = 5, threshold: float = 0.1) -> List[Tuple[str, float]]`

テキストクエリで類似ドキュメントを検索します。

**パラメータ:**
- `query_text` (str): 検索クエリテキスト
- `top_k` (int): 取得する結果数（デフォルト: 5）
- `threshold` (float): 類似度閾値（デフォルト: 0.1）

**戻り値:**
- `List[Tuple[str, float]]`: (ドキュメントキー, 類似度スコア)のリスト

**使用例:**
```python
results = query_engine.search_by_text("徳川家康", top_k=3, threshold=0.3)
for doc_key, score in results:
    print(f"{doc_key}: {score:.4f}")
```

#### `search_by_document(document_key: str, top_k: int = 5, threshold: float = 0.1) -> List[Tuple[str, float]]`

指定されたドキュメントに類似したドキュメントを検索します。

**パラメータ:**
- `document_key` (str): 基準となるドキュメントキー
- `top_k` (int): 取得する結果数（デフォルト: 5）
- `threshold` (float): 類似度閾値（デフォルト: 0.1）

**戻り値:**
- `List[Tuple[str, float]]`: (ドキュメントキー, 類似度スコア)のリスト

**使用例:**
```python
results = query_engine.search_by_document("page-bushou-徳川家康", top_k=5)
for doc_key, score in results:
    print(f"{doc_key}: {score:.4f}")
```

## エラーハンドリング

### 共通例外

システム全体で使用される例外クラス：

```python
class VectorSearchError(Exception):
    """ベクトル検索システムの基底例外クラス"""
    pass

class HTMLProcessingError(VectorSearchError):
    """HTML処理関連のエラー"""
    pass

class VectorEmbeddingError(VectorSearchError):
    """ベクトル埋め込み関連のエラー"""
    pass

class DatabaseError(VectorSearchError):
    """データベース関連のエラー"""
    pass

class SimilarityCalculationError(VectorSearchError):
    """類似度計算関連のエラー"""
    pass
```

### エラーハンドリングの例

```python
from src.html_processor import HTMLProcessor, HTMLProcessingError

try:
    processor = HTMLProcessor()
    text = processor.extract_text("nonexistent.html")
except HTMLProcessingError as e:
    print(f"HTML処理エラー: {e}")
except FileNotFoundError as e:
    print(f"ファイルが見つかりません: {e}")
```

## 設定オプション

### 環境変数による設定

システムの動作は以下の環境変数で制御できます：

```python
import os

# データベース設定
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "data/vectors.db")

# ベクトル埋め込み設定
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# HTML処理設定
HTML_FILE_PATTERN = os.getenv("HTML_FILE_PATTERN", "page-*.html")
HTML_MIN_CONTENT_LENGTH = int(os.getenv("HTML_MIN_CONTENT_LENGTH", "1000"))

# 検索設定
SEARCH_DEFAULT_TOP_K = int(os.getenv("SEARCH_DEFAULT_TOP_K", "5"))
SEARCH_SIMILARITY_THRESHOLD = float(os.getenv("SEARCH_SIMILARITY_THRESHOLD", "0.1"))

# ログ設定
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/system.log")
```

### プログラムでの設定

```python
from config.settings import Settings

# 設定の読み込み
settings = Settings()

# カスタム設定の適用
settings.update({
    "vector_db_path": "custom/vectors.db",
    "embedding_batch_size": 64,
    "log_level": "DEBUG"
})
```