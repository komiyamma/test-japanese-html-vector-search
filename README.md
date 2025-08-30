# 日本語HTMLベクトル検索システム（残念ながら妥当性が低い）

日本語で記述されたHTMLファイル群をベクトル化し、類似度や関係性をスコア化するためのベクトル検索システムです。

## プロジェクト構造

```
japanese-vector-search/
├── src/                    # ソースコード
│   ├── __init__.py
│   ├── html_processor.py   # HTMLファイル処理
│   ├── vector_embedder.py  # ベクトル埋め込み
│   ├── database_manager.py # データベース管理
│   ├── similarity_calculator.py # 類似度計算
│   ├── batch_processor.py  # バッチ処理
│   ├── query_engine.py     # クエリエンジン
│   └── logger.py          # ログ設定
├── config/                 # 設定ファイル
│   ├── __init__.py
│   ├── settings.py        # システム設定
│   ├── config_loader.py   # 設定ファイル読み込み
│   └── config.template.json # 設定テンプレート
├── scripts/               # 実行スクリプト
│   ├── batch_process.py   # バッチ処理スクリプト
│   └── search_cli.py      # 検索CLIスクリプト
├── tests/                 # テストファイル
│   ├── __init__.py
│   ├── test_*.py         # 各コンポーネントのテスト
│   └── test_cli_interfaces.py # CLIインターフェースのテスト
├── data/                  # データファイル（SQLiteデータベース等）
├── logs/                  # ログファイル
├── main.py               # メインエントリーポイント
├── requirements.txt      # 依存関係
└── README.md            # このファイル
```

## インストール

```bash
pip install -r requirements.txt
```

## 使用方法

### メインスクリプト（推奨）

システム全体を管理するメインスクリプトを使用：

```bash
# ヘルプを表示
python main.py --help

# バッチ処理モード（HTMLファイルをベクトル化）
python main.py batch

# 検索モード（インタラクティブ検索）
python main.py search

# バージョン情報を表示
python main.py --version
```

### バッチ処理（直接実行）

HTMLファイルをベクトル化してデータベースに保存：

```bash
# 基本的な使用方法
python scripts/batch_process.py

# 詳細なオプション
python scripts/batch_process.py --help

# 使用例
python scripts/batch_process.py --directory ./html_files --force --log-level DEBUG
python scripts/batch_process.py --dry-run  # 処理対象ファイルのみ表示
python scripts/batch_process.py --batch-size 64 --log-file batch.log
```

#### バッチ処理オプション

- `--directory, -d`: 処理対象ディレクトリ（デフォルト: 現在のディレクトリ）
- `--pattern`: ファイルパターン（デフォルト: page-*.html）
- `--force`: 既存データを強制更新
- `--batch-size`: バッチ処理サイズ（デフォルト: 32）
- `--db-path`: データベースファイルパス
- `--model-name`: 使用するベクトル化モデル
- `--log-level`: ログレベル（DEBUG, INFO, WARNING, ERROR, CRITICAL）
- `--log-file`: ログファイルパス
- `--dry-run`: 実際の処理は行わず、対象ファイルのみ表示

### 検索（直接実行）

ベクトル化されたデータから類似ドキュメントを検索：

```bash
# インタラクティブモード
python scripts/search_cli.py --interactive

# テキストクエリで検索
python scripts/search_cli.py --text "徳川家康"

# ドキュメントキーで類似検索
python scripts/search_cli.py --document "page-bushou-徳川家康"

# 詳細なオプション
python scripts/search_cli.py --help
```

#### 検索オプション

- `--text`: 検索するテキストクエリ
- `--document`: 類似検索の基準となるドキュメントキー
- `--interactive`: インタラクティブモード
- `--top-k`: 取得する結果数（デフォルト: 5）
- `--threshold`: 類似度閾値（デフォルト: 0.1）
- `--db-path`: データベースファイルパス
- `--format`: 出力フォーマット（simple, detailed, json）
- `--log-level`: ログレベル

#### インタラクティブモードのコマンド

```
検索> text 戦国時代              # テキストクエリで検索
検索> doc page-bushou-織田信長   # ドキュメント類似検索
検索> set top-k 10              # 取得結果数を設定
検索> set threshold 0.3         # 類似度閾値を設定
検索> set format detailed       # 出力フォーマットを設定
検索> help                      # ヘルプを表示
検索> quit                      # 終了
```

## 設定

### 設定ファイル

システムの動作は以下の方法で設定できます：

1. **デフォルト設定**: `config/settings.py`
2. **JSON設定ファイル**: `config/config.json` または `config/settings.json`
3. **環境変数**: 各種環境変数で設定を上書き

設定テンプレートファイル（`config/config.template.json`）を参考にして、カスタム設定ファイルを作成できます。

### 環境変数

以下の環境変数で設定を上書きできます：

- `VECTOR_DB_PATH`: データベースファイルパス
- `EMBEDDING_MODEL_NAME`: ベクトル化モデル名
- `EMBEDDING_BATCH_SIZE`: バッチ処理サイズ
- `HTML_FILE_PATTERN`: HTMLファイルパターン
- `HTML_MIN_CONTENT_LENGTH`: 最小コンテンツ長
- `SEARCH_DEFAULT_TOP_K`: デフォルト取得結果数
- `SEARCH_SIMILARITY_THRESHOLD`: 類似度閾値
- `LOG_LEVEL`: ログレベル
- `LOG_FILE`: ログファイルパス

### 設定例

```bash
# 環境変数で設定
export VECTOR_DB_PATH="/custom/path/vectors.db"
export EMBEDDING_BATCH_SIZE=64
export LOG_LEVEL=DEBUG

# カスタム設定ファイルで実行
python scripts/batch_process.py --config custom_config.json
```

## テスト

```bash
# 全テストを実行
python -m pytest

# 特定のテストファイルを実行
python -m pytest tests/test_cli_interfaces.py

# 詳細出力でテスト実行
python -m pytest -v

# カバレッジ付きでテスト実行
python -m pytest --cov=src
```

## トラブルシューティング

### よくある問題

1. **sentence-transformersのインストールエラー**
   ```bash
   pip install --upgrade sentence-transformers torch
   ```

2. **データベースファイルが見つからない**
   - 先にバッチ処理を実行してベクトルデータを作成してください

3. **メモリ不足エラー**
   - バッチサイズを小さくしてください（`--batch-size 16`）

4. **日本語エンコーディングエラー**
   - HTMLファイルがUTF-8でエンコードされていることを確認してください

### ログの確認

```bash
# ログファイルの確認
tail -f logs/system.log

# デバッグレベルでの実行
python main.py batch --log-level DEBUG
```

## 要件

- Python 3.11以上
- 必要なライブラリは requirements.txt を参照
- 推奨: 8GB以上のRAM（大量のHTMLファイル処理時）

## ドキュメント

詳細な情報については、以下のドキュメントを参照してください：

- **[APIリファレンス](docs/api_reference.md)** - 各コンポーネントの詳細なAPI仕様
- **[使用例とサンプルコード](docs/examples.md)** - 実践的な使用例とコードサンプル
- **[トラブルシューティングガイド](docs/troubleshooting.md)** - よくある問題と解決方法

## サンプルコード

### 基本的な使用例

```python
from src.html_processor import HTMLProcessor
from src.vector_embedder import VectorEmbedder
from src.database_manager import DatabaseManager
from src.query_engine import QueryEngine
from src.similarity_calculator import SimilarityCalculator

# コンポーネントの初期化
html_processor = HTMLProcessor()
vector_embedder = VectorEmbedder()
database_manager = DatabaseManager("data/vectors.db")
similarity_calculator = SimilarityCalculator()

# データベーステーブルの作成
database_manager.create_table()

# HTMLファイルの処理
text = html_processor.extract_text("page-bushou-徳川家康.html")
vector = vector_embedder.embed_text(text)
file_key = html_processor.get_file_key("page-bushou-徳川家康.html")
database_manager.store_vector(file_key, vector)

# 検索エンジンの初期化と検索
query_engine = QueryEngine(vector_embedder, database_manager, similarity_calculator)
results = query_engine.search_by_text("戦国時代の武将", top_k=5)

for doc_key, score in results:
    print(f"{doc_key}: {score:.4f}")
```

### 対話型検索の例

```python
# インタラクティブ検索モードの起動
python scripts/search_cli.py --interactive

# 検索コマンドの例
検索> text 徳川家康の生涯
検索> doc page-bushou-織田信長
検索> set top-k 10
検索> set threshold 0.3
```

## パフォーマンス

### 推奨システム要件

- **CPU**: 4コア以上（並列処理時）
- **メモリ**: 8GB以上（大量ファイル処理時）
- **ストレージ**: SSD推奨（データベースアクセス高速化）
- **Python**: 3.8以上

### 処理性能の目安

- **HTMLファイル処理**: 約10-50ファイル/分（ファイルサイズに依存）
- **ベクトル検索**: 1,000ドキュメント中から5件取得で約0.1-0.5秒
- **データベースサイズ**: 1,000ドキュメントで約50-100MB

### 最適化のヒント

```bash
# バッチサイズの調整（メモリ使用量とのトレードオフ）
export EMBEDDING_BATCH_SIZE=64

# 並列処理の有効化
python scripts/batch_process.py --workers 4

# SSDの場合はより大きなバッチサイズが可能
python scripts/batch_process.py --batch-size 128
```

## 開発者向け情報

### テストの実行

```bash
# 全テストの実行
python -m pytest tests/ -v

# 特定のコンポーネントのテスト
python -m pytest tests/test_html_processor.py -v

# カバレッジレポート付きテスト
python -m pytest tests/ --cov=src --cov-report=html

# 統合テストのみ実行
python tests/run_integration_tests.py
```

### コードの品質チェック

```bash
# コードフォーマット
black src/ tests/ scripts/

# リンター
flake8 src/ tests/ scripts/

# 型チェック
mypy src/
```

### 貢献方法

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。
