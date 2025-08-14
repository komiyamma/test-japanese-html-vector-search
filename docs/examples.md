# 使用例とサンプルコード

このドキュメントでは、日本語HTMLベクトル検索システムの具体的な使用例とサンプルコードを紹介します。

## 基本的な使用例

### 1. HTMLファイルの一括処理

```python
#!/usr/bin/env python3
"""
HTMLファイルを一括でベクトル化してデータベースに保存する例
"""

import os
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from src.html_processor import HTMLProcessor
from src.vector_embedder import VectorEmbedder
from src.database_manager import DatabaseManager
from src.batch_processor import BatchProcessor
from src.logger import setup_logger

def main():
    # ログ設定
    logger = setup_logger("batch_example", log_level="INFO")
    
    # コンポーネントの初期化
    html_processor = HTMLProcessor()
    vector_embedder = VectorEmbedder()
    database_manager = DatabaseManager("data/vectors.db")
    
    # データベーステーブルの作成
    database_manager.create_table()
    
    # バッチプロセッサの初期化
    batch_processor = BatchProcessor(
        html_processor=html_processor,
        vector_embedder=vector_embedder,
        database_manager=database_manager
    )
    
    # HTMLファイルの一括処理
    logger.info("HTMLファイルの処理を開始します...")
    result = batch_processor.process_directory(
        directory=".",
        pattern="page-bushou-*.html",
        force=False  # 既存データは更新しない
    )
    
    # 結果の表示
    logger.info(f"処理完了:")
    logger.info(f"  処理されたファイル数: {result['processed_files']}")
    logger.info(f"  スキップされたファイル数: {result['skipped_files']}")
    logger.info(f"  エラー数: {result['errors']}")
    
    if result['error_details']:
        logger.warning("エラーの詳細:")
        for error in result['error_details']:
            logger.warning(f"  {error}")

if __name__ == "__main__":
    main()
```

### 2. テキストクエリによる検索

```python
#!/usr/bin/env python3
"""
テキストクエリで類似ドキュメントを検索する例
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.vector_embedder import VectorEmbedder
from src.database_manager import DatabaseManager
from src.similarity_calculator import SimilarityCalculator
from src.query_engine import QueryEngine
from src.logger import setup_logger

def search_example():
    # ログ設定
    logger = setup_logger("search_example", log_level="INFO")
    
    # コンポーネントの初期化
    vector_embedder = VectorEmbedder()
    database_manager = DatabaseManager("data/vectors.db")
    similarity_calculator = SimilarityCalculator()
    
    # クエリエンジンの初期化
    query_engine = QueryEngine(
        vector_embedder=vector_embedder,
        database_manager=database_manager,
        similarity_calculator=similarity_calculator
    )
    
    # 検索クエリのリスト
    queries = [
        "戦国時代の武将",
        "江戸幕府の創設者",
        "本能寺の変",
        "関ヶ原の戦い",
        "天下統一"
    ]
    
    for query in queries:
        logger.info(f"\n検索クエリ: '{query}'")
        logger.info("-" * 50)
        
        try:
            # テキストクエリで検索
            results = query_engine.search_by_text(
                query_text=query,
                top_k=3,
                threshold=0.2
            )
            
            if results:
                for i, (doc_key, score) in enumerate(results, 1):
                    logger.info(f"{i}. {doc_key} (類似度: {score:.4f})")
            else:
                logger.info("該当するドキュメントが見つかりませんでした。")
                
        except Exception as e:
            logger.error(f"検索エラー: {e}")

if __name__ == "__main__":
    search_example()
```

### 3. ドキュメント類似検索

```python
#!/usr/bin/env python3
"""
特定のドキュメントに類似したドキュメントを検索する例
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.vector_embedder import VectorEmbedder
from src.database_manager import DatabaseManager
from src.similarity_calculator import SimilarityCalculator
from src.query_engine import QueryEngine
from src.logger import setup_logger

def document_similarity_example():
    # ログ設定
    logger = setup_logger("doc_similarity_example", log_level="INFO")
    
    # コンポーネントの初期化
    vector_embedder = VectorEmbedder()
    database_manager = DatabaseManager("data/vectors.db")
    similarity_calculator = SimilarityCalculator()
    
    # クエリエンジンの初期化
    query_engine = QueryEngine(
        vector_embedder=vector_embedder,
        database_manager=database_manager,
        similarity_calculator=similarity_calculator
    )
    
    # 基準となるドキュメントのリスト
    reference_documents = [
        "page-bushou-徳川家康",
        "page-bushou-織田信長",
        "page-bushou-豊臣秀吉",
        "page-bushou-武田信玄"
    ]
    
    for doc_key in reference_documents:
        logger.info(f"\n基準ドキュメント: '{doc_key}'")
        logger.info("-" * 60)
        
        try:
            # ドキュメント類似検索
            results = query_engine.search_by_document(
                document_key=doc_key,
                top_k=5,
                threshold=0.1
            )
            
            if results:
                logger.info("類似ドキュメント:")
                for i, (similar_doc, score) in enumerate(results, 1):
                    if similar_doc != doc_key:  # 自分自身は除外
                        logger.info(f"{i}. {similar_doc} (類似度: {score:.4f})")
            else:
                logger.info("類似ドキュメントが見つかりませんでした。")
                
        except Exception as e:
            logger.error(f"検索エラー: {e}")

if __name__ == "__main__":
    document_similarity_example()
```

## 高度な使用例

### 4. カスタム設定での処理

```python
#!/usr/bin/env python3
"""
カスタム設定を使用した処理の例
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.html_processor import HTMLProcessor
from src.vector_embedder import VectorEmbedder
from src.database_manager import DatabaseManager
from src.batch_processor import BatchProcessor
from config.config_loader import ConfigLoader
from src.logger import setup_logger

def custom_config_example():
    # カスタム設定の読み込み
    config_loader = ConfigLoader()
    config = config_loader.load_config("config/custom_config.json")
    
    # ログ設定
    logger = setup_logger(
        "custom_example", 
        log_level=config.get("log_level", "INFO"),
        log_file=config.get("log_file", "logs/custom.log")
    )
    
    # カスタム設定でコンポーネントを初期化
    html_processor = HTMLProcessor()
    
    # カスタムモデルでベクトル埋め込み
    vector_embedder = VectorEmbedder(
        model_name=config.get("embedding_model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    )
    
    # カスタムデータベースパス
    database_manager = DatabaseManager(
        db_path=config.get("vector_db_path", "data/vectors.db")
    )
    database_manager.create_table()
    
    # バッチプロセッサの初期化
    batch_processor = BatchProcessor(
        html_processor=html_processor,
        vector_embedder=vector_embedder,
        database_manager=database_manager
    )
    
    # カスタム設定での処理
    result = batch_processor.process_directory(
        directory=config.get("html_directory", "."),
        pattern=config.get("html_file_pattern", "page-*.html"),
        force=config.get("force_update", False)
    )
    
    logger.info(f"カスタム設定での処理完了: {result}")

if __name__ == "__main__":
    custom_config_example()
```

### 5. エラーハンドリングと再試行

```python
#!/usr/bin/env python3
"""
エラーハンドリングと再試行機能の例
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from src.vector_embedder import VectorEmbedder
from src.database_manager import DatabaseManager
from src.similarity_calculator import SimilarityCalculator
from src.query_engine import QueryEngine
from src.logger import setup_logger

class RobustQueryEngine:
    """エラーハンドリングと再試行機能を持つクエリエンジン"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = setup_logger("robust_query", log_level="INFO")
        
        # コンポーネントの初期化
        self.vector_embedder = VectorEmbedder()
        self.database_manager = DatabaseManager("data/vectors.db")
        self.similarity_calculator = SimilarityCalculator()
        self.query_engine = QueryEngine(
            vector_embedder=self.vector_embedder,
            database_manager=self.database_manager,
            similarity_calculator=self.similarity_calculator
        )
    
    def search_with_retry(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """再試行機能付きの検索"""
        
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"検索試行 {attempt + 1}/{self.max_retries}: '{query_text}'")
                
                results = self.query_engine.search_by_text(
                    query_text=query_text,
                    top_k=top_k,
                    threshold=0.1
                )
                
                self.logger.info(f"検索成功: {len(results)}件の結果")
                return results
                
            except Exception as e:
                self.logger.warning(f"検索試行 {attempt + 1} 失敗: {e}")
                
                if attempt < self.max_retries - 1:
                    self.logger.info(f"{self.retry_delay}秒後に再試行します...")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"最大試行回数に達しました。検索を中止します。")
                    raise
        
        return []

def robust_search_example():
    """堅牢な検索の例"""
    
    robust_engine = RobustQueryEngine(max_retries=3, retry_delay=2.0)
    
    queries = [
        "徳川家康の生涯",
        "戦国時代の合戦",
        "江戸時代の政治",
        "明治維新の影響"
    ]
    
    for query in queries:
        try:
            results = robust_engine.search_with_retry(query, top_k=3)
            
            print(f"\n検索結果 - '{query}':")
            print("-" * 50)
            
            if results:
                for i, (doc_key, score) in enumerate(results, 1):
                    print(f"{i}. {doc_key} (類似度: {score:.4f})")
            else:
                print("結果が見つかりませんでした。")
                
        except Exception as e:
            print(f"検索エラー: {e}")

if __name__ == "__main__":
    robust_search_example()
```

### 6. バッチ処理の進捗監視

```python
#!/usr/bin/env python3
"""
バッチ処理の進捗を詳細に監視する例
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from src.html_processor import HTMLProcessor
from src.vector_embedder import VectorEmbedder
from src.database_manager import DatabaseManager
from src.logger import setup_logger

class ProgressMonitor:
    """進捗監視クラス"""
    
    def __init__(self):
        self.logger = setup_logger("progress_monitor", log_level="INFO")
        self.start_time = None
        self.processed_count = 0
        self.total_count = 0
    
    def start(self, total_count: int):
        """監視開始"""
        self.start_time = time.time()
        self.processed_count = 0
        self.total_count = total_count
        self.logger.info(f"処理開始: 総ファイル数 {total_count}")
    
    def update(self, processed_count: int):
        """進捗更新"""
        self.processed_count = processed_count
        
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            progress_percent = (processed_count / self.total_count) * 100
            
            if processed_count > 0:
                avg_time_per_file = elapsed_time / processed_count
                estimated_remaining = avg_time_per_file * (self.total_count - processed_count)
                
                self.logger.info(
                    f"進捗: {processed_count}/{self.total_count} "
                    f"({progress_percent:.1f}%) "
                    f"経過時間: {elapsed_time:.1f}秒 "
                    f"推定残り時間: {estimated_remaining:.1f}秒"
                )
    
    def finish(self):
        """監視終了"""
        if self.start_time:
            total_time = time.time() - self.start_time
            self.logger.info(f"処理完了: 総時間 {total_time:.1f}秒")

def monitored_batch_processing():
    """進捗監視付きバッチ処理の例"""
    
    # コンポーネントの初期化
    html_processor = HTMLProcessor()
    vector_embedder = VectorEmbedder()
    database_manager = DatabaseManager("data/vectors.db")
    database_manager.create_table()
    
    # 進捗監視の初期化
    monitor = ProgressMonitor()
    
    # 処理対象ファイルの取得
    html_files = list(Path(".").glob("page-bushou-*.html"))
    
    # 監視開始
    monitor.start(len(html_files))
    
    processed_files = []
    error_files = []
    
    for i, html_file in enumerate(html_files):
        try:
            # HTMLファイルの処理
            text = html_processor.extract_text(str(html_file))
            
            # ベクトル化
            vector = vector_embedder.embed_text(text)
            
            # データベースに保存
            file_key = html_processor.get_file_key(str(html_file))
            database_manager.store_vector(file_key, vector)
            
            processed_files.append(str(html_file))
            
        except Exception as e:
            error_files.append((str(html_file), str(e)))
            monitor.logger.error(f"ファイル処理エラー {html_file}: {e}")
        
        # 進捗更新
        monitor.update(i + 1)
    
    # 監視終了
    monitor.finish()
    
    # 結果サマリー
    monitor.logger.info(f"処理結果:")
    monitor.logger.info(f"  成功: {len(processed_files)}ファイル")
    monitor.logger.info(f"  エラー: {len(error_files)}ファイル")
    
    if error_files:
        monitor.logger.warning("エラーファイル:")
        for file_path, error in error_files:
            monitor.logger.warning(f"  {file_path}: {error}")

if __name__ == "__main__":
    monitored_batch_processing()
```

## インタラクティブな使用例

### 7. 対話型検索インターフェース

```python
#!/usr/bin/env python3
"""
対話型検索インターフェースの例
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.vector_embedder import VectorEmbedder
from src.database_manager import DatabaseManager
from src.similarity_calculator import SimilarityCalculator
from src.query_engine import QueryEngine

class InteractiveSearch:
    """対話型検索クラス"""
    
    def __init__(self):
        # コンポーネントの初期化
        self.vector_embedder = VectorEmbedder()
        self.database_manager = DatabaseManager("data/vectors.db")
        self.similarity_calculator = SimilarityCalculator()
        self.query_engine = QueryEngine(
            vector_embedder=self.vector_embedder,
            database_manager=self.database_manager,
            similarity_calculator=self.similarity_calculator
        )
        
        # 設定
        self.top_k = 5
        self.threshold = 0.1
    
    def show_help(self):
        """ヘルプを表示"""
        print("\n=== 日本語HTMLベクトル検索システム ===")
        print("コマンド:")
        print("  text <クエリ>     - テキストクエリで検索")
        print("  doc <ドキュメント> - ドキュメント類似検索")
        print("  set top-k <数値>  - 取得結果数を設定")
        print("  set threshold <値> - 類似度閾値を設定")
        print("  show settings     - 現在の設定を表示")
        print("  help             - このヘルプを表示")
        print("  quit             - 終了")
        print()
    
    def show_settings(self):
        """現在の設定を表示"""
        print(f"現在の設定:")
        print(f"  取得結果数: {self.top_k}")
        print(f"  類似度閾値: {self.threshold}")
        print()
    
    def search_by_text(self, query_text: str):
        """テキスト検索"""
        try:
            results = self.query_engine.search_by_text(
                query_text=query_text,
                top_k=self.top_k,
                threshold=self.threshold
            )
            
            if results:
                print(f"\n検索結果 ('{query_text}'):")
                print("-" * 50)
                for i, (doc_key, score) in enumerate(results, 1):
                    print(f"{i}. {doc_key} (類似度: {score:.4f})")
            else:
                print("該当するドキュメントが見つかりませんでした。")
            print()
            
        except Exception as e:
            print(f"検索エラー: {e}\n")
    
    def search_by_document(self, document_key: str):
        """ドキュメント類似検索"""
        try:
            results = self.query_engine.search_by_document(
                document_key=document_key,
                top_k=self.top_k,
                threshold=self.threshold
            )
            
            if results:
                print(f"\n類似ドキュメント ('{document_key}'):")
                print("-" * 50)
                for i, (doc_key, score) in enumerate(results, 1):
                    if doc_key != document_key:  # 自分自身は除外
                        print(f"{i}. {doc_key} (類似度: {score:.4f})")
            else:
                print("類似ドキュメントが見つかりませんでした。")
            print()
            
        except Exception as e:
            print(f"検索エラー: {e}\n")
    
    def run(self):
        """対話型検索を実行"""
        self.show_help()
        
        while True:
            try:
                user_input = input("検索> ").strip()
                
                if not user_input:
                    continue
                
                parts = user_input.split()
                command = parts[0].lower()
                
                if command == "quit":
                    print("検索を終了します。")
                    break
                
                elif command == "help":
                    self.show_help()
                
                elif command == "show" and len(parts) > 1 and parts[1] == "settings":
                    self.show_settings()
                
                elif command == "text" and len(parts) > 1:
                    query_text = " ".join(parts[1:])
                    self.search_by_text(query_text)
                
                elif command == "doc" and len(parts) > 1:
                    document_key = parts[1]
                    self.search_by_document(document_key)
                
                elif command == "set" and len(parts) == 3:
                    setting_name = parts[1]
                    setting_value = parts[2]
                    
                    if setting_name == "top-k":
                        try:
                            self.top_k = int(setting_value)
                            print(f"取得結果数を {self.top_k} に設定しました。\n")
                        except ValueError:
                            print("無効な数値です。\n")
                    
                    elif setting_name == "threshold":
                        try:
                            self.threshold = float(setting_value)
                            print(f"類似度閾値を {self.threshold} に設定しました。\n")
                        except ValueError:
                            print("無効な数値です。\n")
                    
                    else:
                        print("無効な設定項目です。\n")
                
                else:
                    print("無効なコマンドです。'help' でヘルプを表示します。\n")
            
            except KeyboardInterrupt:
                print("\n検索を終了します。")
                break
            except Exception as e:
                print(f"エラー: {e}\n")

if __name__ == "__main__":
    interactive_search = InteractiveSearch()
    interactive_search.run()
```

## 設定ファイルの例

### カスタム設定ファイル (custom_config.json)

```json
{
  "vector_db_path": "data/custom_vectors.db",
  "embedding_model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "embedding_batch_size": 64,
  "html_directory": "./html_files",
  "html_file_pattern": "page-bushou-*.html",
  "html_min_content_length": 500,
  "search_default_top_k": 10,
  "search_similarity_threshold": 0.15,
  "log_level": "DEBUG",
  "log_file": "logs/custom.log",
  "force_update": false
}
```

これらの例を参考に、プロジェクトの要件に応じてカスタマイズしてください。各例は独立して実行可能で、システムの様々な機能を実際に試すことができます。