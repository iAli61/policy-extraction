import unittest
from src.markdown_processor import MarkdownProcessor

class TestMarkdownProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = MarkdownProcessor()

    def test_parsing_simple_markdown(self):
        markdown_text = "# Title\n\nThis is a simple markdown text."
        chunks = self.processor.parse(markdown_text)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Title", chunks[0])
        self.assertIn("This is a simple markdown text.", chunks[0])

    def test_parsing_markdown_with_table(self):
        markdown_text = "| Header1 | Header2 |\n|---------|---------|\n| Row1   | Data1   |\n| Row2   | Data2   |"
        chunks = self.processor.parse(markdown_text)
        self.assertGreater(len(chunks), 0)
        self.assertIn("Header1", chunks[0])
        self.assertIn("Row1", chunks[0])
        self.assertIn("Row2", chunks[0])

    def test_chunking_large_table(self):
        markdown_text = "| Header1 | Header2 |\n|---------|---------|\n" + \
                        "| Row1   | Data1   |\n" * 20  # Simulate a large table
        chunks = self.processor.parse(markdown_text)
        self.assertGreater(len(chunks), 1)
        self.assertIn("Header1", chunks[0])
        self.assertIn("Row1", chunks[0])
        self.assertIn("Row2", chunks[1])  # Check that subsequent rows are in the next chunk

    def test_parsing_invalid_markdown(self):
        markdown_text = "This is not a valid markdown."
        chunks = self.processor.parse(markdown_text)
        self.assertEqual(len(chunks), 1)
        self.assertIn("This is not a valid markdown.", chunks[0])

if __name__ == '__main__':
    unittest.main()