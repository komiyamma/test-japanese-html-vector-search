# ドキュメント

日本語HTMLベクトル検索システムの詳細ドキュメントです。

## ドキュメント一覧

### 📚 [APIリファレンス](api_reference.md)
各コンポーネントの詳細なAPI仕様とメソッドの説明

- HTMLProcessor - HTMLファイルの処理
- VectorEmbedder - テキストのベクトル化
- DatabaseManager - SQLiteデータベース管理
- SimilarityCalculator - 類似度計算
- BatchProcessor - バッチ処理
- QueryEngine - 検索エンジン

### 💡 [使用例とサンプルコード](examples.md)
実践的な使用例とコードサンプル

- 基本的な使用例
- HTMLファイルの一括処理
- テキストクエリによる検索
- ドキュメント類似検索
- カスタム設定での処理
- エラーハンドリングと再試行
- バッチ処理の進捗監視
- 対話型検索インターフェース

### 🔧 [トラブルシューティングガイド](troubleshooting.md)
よくある問題と解決方法

- インストール関連の問題
- 実行時エラー
- 検索関連の問題
- ログとデバッグ
- パフォーマンス最適化

## クイックスタート

### 1. インストール

```bash
# 依存関係のインストール
pip install -r requirements.txt

# データベースディレクトリの作成
mkdir -p data logs
```

### 2. HTMLファイルの処理

```bash
# バッチ処理でHTMLファイルをベクトル化
python main.py batch

# または直接スクリプトを実行
python scripts/batch_process.py
```

### 3. 検索の実行

```bash
# インタラクティブ検索モード
python main.py search

# または直接検索
python scripts/search_cli.py --text "徳川家康"
```

## システム構成

```
日本語HTMLベクトル検索システム
├── HTMLファイル読み込み (HTMLProcessor)
├── テキスト抽出・前処理
├── ベクトル化 (VectorEmbedder)
├── データベース保存 (DatabaseManager)
├── 類似度計算 (SimilarityCalculator)
└── 検索インターフェース (QueryEngine)
```

## 主要機能

- ✅ 日本語HTMLファイルの自動処理
- ✅ Sentence Transformersによる高精度ベクトル化
- ✅ SQLiteによる効率的なデータ管理
- ✅ コサイン類似度による検索
- ✅ バッチ処理とプログレス表示
- ✅ 対話型検索インターフェース
- ✅ 設定ファイルによるカスタマイズ
- ✅ 包括的なエラーハンドリング
- ✅ 詳細なログ出力

## サポート

問題が発生した場合は、以下の順序で確認してください：

1. **[トラブルシューティングガイド](troubleshooting.md)** で類似の問題を確認
2. **ログファイル** (`logs/system.log`) でエラーの詳細を確認
3. **システム診断スクリプト** を実行して環境を確認
4. **[使用例](examples.md)** で正しい使用方法を確認

### システム診断

```bash
# システム診断の実行
python -c "
import sys
sys.path.append('docs')
from troubleshooting import system_diagnosis
system_diagnosis()
"
```

## 更新履歴

- **v1.0.0** - 初期リリース
  - 基本的なHTMLファイル処理機能
  - ベクトル化と検索機能
  - バッチ処理とCLIインターフェース
  - 包括的なテストスイート
  - 詳細なドキュメント

---

詳細な情報については、各ドキュメントを参照してください。