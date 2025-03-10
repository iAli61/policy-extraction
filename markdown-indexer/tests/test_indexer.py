import unittest
from src.indexer import Indexer
from src.markdown_processor import MarkdownProcessor
from src.embeddings import EmbeddingGenerator
from src.storage.faiss_storage import FaissStorage
from src.storage.azure_search import AzureSearchStorage

class TestIndexer(unittest.TestCase):

    def setUp(self):
        self.markdown_text = """
        # Sample Document

        This is a sample markdown document.

        | Header 1 | Header 2 |
        |----------|----------|
        | Row 1    | Data 1   |
        | Row 2    | Data 2   |

        ## Another Section

        More text here.
        """
        self.processor = MarkdownProcessor()
        self.embedding_generator = EmbeddingGenerator(model_name="distilbert-base-uncased")
        self.faiss_storage = FaissStorage()
        self.azure_storage = AzureSearchStorage()
        self.indexer = Indexer(processor=self.processor, storage=self.faiss_storage)

    def test_indexing_markdown(self):
        chunks = self.processor.chunk_markdown(self.markdown_text)
        embeddings = self.embedding_generator.generate_embeddings(chunks)
        for chunk, embedding in zip(chunks, embeddings):
            self.indexer.index(chunk, embedding)
        
        self.assertEqual(len(chunks), len(embeddings))
        self.assertTrue(self.indexer.storage.count() > 0)

    def test_indexing_with_azure(self):
        self.indexer.storage = self.azure_storage
        chunks = self.processor.chunk_markdown(self.markdown_text)
        embeddings = self.embedding_generator.generate_embeddings(chunks)
        for chunk, embedding in zip(chunks, embeddings):
            self.indexer.index(chunk, embedding)
        
        self.assertEqual(len(chunks), len(embeddings))
        self.assertTrue(self.indexer.storage.count() > 0)

if __name__ == '__main__':
    unittest.main()