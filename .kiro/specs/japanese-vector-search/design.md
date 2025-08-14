# 設計書

## 概要

日本語HTMLファイルのベクトル検索システムは、HTMLコンテンツの抽出、ベクトル化、SQLiteでの保存、類似度検索を行うPythonベースのシステムです。Sentence Transformersライブラリを使用して日本語テキストをベクトル化し、効率的な検索機能を提供します。

## アーキテクチャ

### システム構成図

```
[HTMLファイル群] → [HTMLパーサー] → [テキスト抽出] → [ベクトル化エンジン] → [SQLiteデータベース]
                                                                ↓
[クエリインターフェース] ← [類似度計算エンジン] ← [ベクトル検索エンジン]
```

### 主要コンポーネント

1. **HTMLProcessor**: HTMLファイルの読み込みとテキスト抽出
2. **VectorEmbedder**: 日本語テキストのベクトル化
3. **DatabaseManager**: SQLiteでのベクトルデータ管理
4. **SimilarityCalculator**: コサイン類似度計算
5. **BatchProcessor**: バッチ処理とプログレス管理
6. **QueryEngine**: 検索クエリの処理

## コンポーネントとインターフェース

### HTMLProcessor
```python
class HTMLProcessor:
    def extract_text(self, html_file_path: str) -> str
    def get_file_key(self, html_file_path: str) -> str
    def validate_content_length(self, text: str) -> bool
```

### VectorEmbedder
```python
class VectorEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    def embed_text(self, text: str) -> np.ndarray
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]
    def preprocess_japanese_text(self, text: str) -> str
    def extract_named_entities(self, text: str) -> List[str]
```

### EnhancedSimilarityCalculator
```python
class EnhancedSimilarityCalculator:
    def __init__(self, use_weighted_similarity: bool = True)
    def cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float
    def weighted_similarity(self, query_text: str, doc_text: str, 
                          vector_similarity: float) -> float
    def find_similar_documents_enhanced(self, query_text: str, query_vector: np.ndarray,
                                      document_vectors: Dict[str, np.ndarray],
                                      document_texts: Dict[str, str],
                                      top_k: int = 5) -> List[Tuple[str, float]]
```

### DatabaseManager
```python
class DatabaseManager:
    def __init__(self, db_path: str = "vectors.db")
    def create_table(self) -> None
    def store_vector(self, key: str, vector: np.ndarray) -> None
    def get_vector(self, key: str) -> Optional[np.ndarray]
    def get_all_vectors(self) -> Dict[str, np.ndarray]
    def update_vector(self, key: str, vector: np.ndarray) -> None
```

### SimilarityCalculator
```python
class SimilarityCalculator:
    def cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float
    def find_similar_documents(self, query_vector: np.ndarray, 
                             document_vectors: Dict[str, np.ndarray], 
                             top_k: int = 5) -> List[Tuple[str, float]]
```

## データモデル

### SQLiteテーブル構造

```sql
CREATE TABLE IF NOT EXISTS document_vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_key TEXT UNIQUE NOT NULL,
    vector_data BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_document_key ON document_vectors(document_key);
```

### ベクトルデータ形式
- NumPy配列をpickle形式でシリアライズしてBLOB型で保存
- ベクトル次元数: 384次元（使用モデルに依存）
- データ型: float32（メモリ効率とパフォーマンスのバランス）

## エラーハンドリング

### HTMLファイル処理エラー
- ファイル読み込みエラー: ログ出力して次のファイルに進む
- エンコーディングエラー: UTF-8で再試行、失敗時はスキップ
- 空コンテンツ: 警告ログを出力して処理継続

### データベースエラー
- 接続エラー: 再試行機構（最大3回）
- 重複キーエラー: UPDATE文で既存レコードを更新
- ディスク容量不足: 明確なエラーメッセージで処理停止

### ベクトル化エラー
- モデル読み込みエラー: システム終了（必須コンポーネント）
- メモリ不足: バッチサイズを動的に調整
- 長すぎるテキスト: チャンク分割して処理

## テスト戦略

### 単体テスト
- 各コンポーネントの個別機能テスト
- エッジケース（空文字列、巨大ファイル、特殊文字）のテスト
- エラーハンドリングのテスト

### 統合テスト
- HTMLファイルからベクトル保存までの全体フロー
- データベースの整合性テスト
- バッチ処理の正確性テスト

### パフォーマンステスト
- 大量ファイル処理時のメモリ使用量
- 類似度検索の応答時間
- データベースクエリのパフォーマンス

### テストデータ
- 小さなサンプルHTMLファイル（1,000文字未満）
- 中規模ファイル（10,000文字程度）
- 大規模ファイル（30,000文字程度）
- 日本語特有の文字（ひらがな、カタカナ、漢字、記号）を含むテストケース
- 歴史人物関連性テストケース（織田信長→豊臣秀吉、徳川家康など）

## ベクトル化とスコアリングの改善戦略

### 問題分析
現在のシステムで「織田信長」検索時に関連人物が高スコアにならない原因：

1. **モデルの限界**: 使用中のparaphrase-multilingual-MiniLM-L12-v2は一般的な多言語モデルで、日本史の専門知識が不足
2. **単純なコサイン類似度**: ベクトル間の角度のみを考慮し、歴史的文脈や固有名詞の重要度を考慮していない
3. **テキスト前処理不足**: 日本語固有名詞の正規化や重要語句の強調が不十分

### 改善アプローチ

#### 1. マルチモデル戦略
```python
class MultiModelEmbedder:
    def __init__(self):
        self.general_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.japanese_model = SentenceTransformer('sonoisa/sentence-bert-base-ja-mean-tokens-v2')
    
    def embed_with_ensemble(self, text: str) -> np.ndarray:
        # 複数モデルの結果を組み合わせ
        pass
```

#### 2. 重み付き類似度計算
```python
def calculate_weighted_similarity(self, query: str, document: str, vector_sim: float) -> float:
    # 固有名詞マッチングボーナス
    named_entity_bonus = self.calculate_named_entity_overlap(query, document)
    
    # 歴史的関連性ボーナス
    historical_bonus = self.calculate_historical_relationship_bonus(query, document)
    
    # 最終スコア = ベクトル類似度 + 重み付きボーナス
    return vector_sim * 0.7 + named_entity_bonus * 0.2 + historical_bonus * 0.1
```

#### 3. 日本語テキスト前処理強化
```python
def preprocess_japanese_text(self, text: str) -> str:
    # 歴史人物名の正規化
    text = self.normalize_historical_names(text)
    
    # 重要語句の強調
    text = self.emphasize_important_terms(text)
    
    return text
```