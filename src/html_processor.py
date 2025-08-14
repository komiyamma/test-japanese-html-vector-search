"""
HTMLファイル処理クラス
HTML file processing class

日本語HTMLファイルからテキストを抽出し、適切なキーを生成する機能を提供します。
"""

import os
import logging
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup


class HTMLProcessor:
    """
    HTMLファイルの処理を行うクラス
    
    主な機能:
    - HTMLファイルからテキスト抽出（UTF-8エンコーディング）
    - ファイル名からキー生成
    - コンテンツ長の検証
    
    注意: HTMLファイルはUTF-8エンコーディングで保存されていることが前提です。
    """
    
    def __init__(self):
        """HTMLProcessorを初期化します"""
        self.logger = logging.getLogger(__name__)
    
    def extract_text(self, html_file_path: str) -> str:
        """
        HTMLファイルからテキストを抽出します
        
        Args:
            html_file_path (str): HTMLファイルのパス
            
        Returns:
            str: 抽出されたテキスト
            
        Raises:
            FileNotFoundError: ファイルが見つからない場合
            UnicodeDecodeError: エンコーディングエラーの場合
        """
        if not os.path.exists(html_file_path):
            raise FileNotFoundError(f"HTMLファイルが見つかりません: {html_file_path}")
        
        # HTMLファイルはUTF-8で保証されているため、UTF-8で読み込み
        try:
            with open(html_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            self.logger.debug(f"ファイル {html_file_path} をUTF-8で読み込みました")
        except UnicodeDecodeError as e:
            self.logger.error(f"UTF-8でファイルを読み込めませんでした: {html_file_path}")
            raise
        
        # BeautifulSoupでHTMLを解析してテキストを抽出
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # スクリプトとスタイルタグを除去
            for script in soup(["script", "style"]):
                script.decompose()
            
            # テキストを抽出し、余分な空白を除去
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            self.logger.error(f"HTMLの解析中にエラーが発生しました: {html_file_path}, エラー: {e}")
            raise
    
    def get_file_key(self, html_file_path: str) -> str:
        """
        HTMLファイルのパスからキーを生成します
        
        Args:
            html_file_path (str): HTMLファイルのパス
            
        Returns:
            str: 拡張子を除いたファイル名をキーとして返す
        """
        file_path = Path(html_file_path)
        return file_path.stem
    
    def validate_content_length(self, text: str) -> bool:
        """
        コンテンツ長を検証し、適切な警告を出力します
        
        Args:
            text (str): 検証するテキスト
            
        Returns:
            bool: 処理を続行すべきかどうか（常にTrue、警告のみ出力）
        """
        text_length = len(text)
        
        if text_length < 1000:
            self.logger.warning(
                f"コンテンツが短すぎます（{text_length}文字）。"
                f"1,000文字未満のファイルですが、処理を続行します。"
            )
        elif text_length > 30000:
            self.logger.info(
                f"大きなコンテンツを処理中です（{text_length}文字）。"
                f"メモリ使用量に注意してください。"
            )
        else:
            self.logger.debug(f"適切なコンテンツ長です: {text_length}文字")
        
        return True