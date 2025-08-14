"""
HTMLProcessorクラスの単体テスト
Unit tests for HTMLProcessor class
"""

import os
import tempfile
import unittest
from unittest.mock import patch, mock_open
from pathlib import Path

from src.html_processor import HTMLProcessor


class TestHTMLProcessor(unittest.TestCase):
    """HTMLProcessorクラスのテストケース"""
    
    def setUp(self):
        """テストの前準備"""
        self.processor = HTMLProcessor()
    
    def test_extract_text_basic_html(self):
        """基本的なHTMLからのテキスト抽出をテスト"""
        html_content = """
        <html>
        <head><title>テストページ</title></head>
        <body>
            <h1>見出し</h1>
            <p>これは日本語のテストコンテンツです。</p>
            <script>console.log('script');</script>
            <style>body { color: red; }</style>
        </body>
        </html>
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html_content)
            temp_file = f.name
        
        try:
            result = self.processor.extract_text(temp_file)
            self.assertIn("見出し", result)
            self.assertIn("これは日本語のテストコンテンツです。", result)
            self.assertNotIn("console.log", result)  # スクリプトは除去される
            self.assertNotIn("color: red", result)  # スタイルは除去される
        finally:
            os.unlink(temp_file)
    
    def test_extract_text_file_not_found(self):
        """存在しないファイルの処理をテスト"""
        with self.assertRaises(FileNotFoundError):
            self.processor.extract_text("non_existent_file.html")
    
    def test_extract_text_utf8_content(self):
        """UTF-8エンコードされたHTMLコンテンツの処理をテスト"""
        html_content = "<html><body>日本語テスト UTF-8</body></html>"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html_content)
            temp_file = f.name
        
        try:
            result = self.processor.extract_text(temp_file)
            self.assertIn("日本語テスト UTF-8", result)
        finally:
            os.unlink(temp_file)
    
    @patch('builtins.open', side_effect=UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid'))
    def test_extract_text_encoding_error(self, mock_file):
        """UTF-8エンコーディングで失敗した場合のテスト"""
        with patch('os.path.exists', return_value=True):
            with self.assertRaises(UnicodeDecodeError):
                self.processor.extract_text("test.html")
    
    def test_get_file_key_basic(self):
        """基本的なファイルキー生成をテスト"""
        test_cases = [
            ("page-bushou-羽柴秀吉.html", "page-bushou-羽柴秀吉"),
            ("page-bushou-徳川家康.html", "page-bushou-徳川家康"),
            ("/path/to/page-bushou-織田信長.html", "page-bushou-織田信長"),
            ("simple.html", "simple"),
        ]
        
        for file_path, expected_key in test_cases:
            with self.subTest(file_path=file_path):
                result = self.processor.get_file_key(file_path)
                self.assertEqual(result, expected_key)
    
    def test_get_file_key_no_extension(self):
        """拡張子がないファイルのキー生成をテスト"""
        result = self.processor.get_file_key("filename_without_extension")
        self.assertEqual(result, "filename_without_extension")
    
    def test_validate_content_length_short(self):
        """短いコンテンツの検証をテスト"""
        with patch.object(self.processor.logger, 'warning') as mock_warning:
            short_text = "短い" * 100  # 200文字程度
            result = self.processor.validate_content_length(short_text)
            
            self.assertTrue(result)
            mock_warning.assert_called_once()
            self.assertIn("コンテンツが短すぎます", mock_warning.call_args[0][0])
    
    def test_validate_content_length_long(self):
        """長いコンテンツの検証をテスト"""
        with patch.object(self.processor.logger, 'info') as mock_info:
            long_text = "長い" * 20000  # 40,000文字程度
            result = self.processor.validate_content_length(long_text)
            
            self.assertTrue(result)
            mock_info.assert_called_once()
            self.assertIn("大きなコンテンツを処理中", mock_info.call_args[0][0])
    
    def test_validate_content_length_normal(self):
        """適切な長さのコンテンツの検証をテスト"""
        with patch.object(self.processor.logger, 'debug') as mock_debug:
            normal_text = "適切な長さ" * 500  # 5,000文字程度
            result = self.processor.validate_content_length(normal_text)
            
            self.assertTrue(result)
            mock_debug.assert_called_once()
            self.assertIn("適切なコンテンツ長", mock_debug.call_args[0][0])
    
    def test_extract_text_with_japanese_characters(self):
        """日本語文字（ひらがな、カタカナ、漢字）を含むHTMLのテスト"""
        html_content = """
        <html>
        <body>
            <h1>日本の歴史</h1>
            <p>ひらがな：あいうえお</p>
            <p>カタカナ：アイウエオ</p>
            <p>漢字：日本語処理</p>
            <p>記号：！？（）「」</p>
        </body>
        </html>
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html_content)
            temp_file = f.name
        
        try:
            result = self.processor.extract_text(temp_file)
            self.assertIn("日本の歴史", result)
            self.assertIn("あいうえお", result)
            self.assertIn("アイウエオ", result)
            self.assertIn("日本語処理", result)
            self.assertIn("！？（）「」", result)
        finally:
            os.unlink(temp_file)
    
    def test_extract_text_empty_html(self):
        """空のHTMLファイルの処理をテスト"""
        html_content = "<html><body></body></html>"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html_content)
            temp_file = f.name
        
        try:
            result = self.processor.extract_text(temp_file)
            self.assertEqual(result.strip(), "")
        finally:
            os.unlink(temp_file)


if __name__ == '__main__':
    unittest.main()