import pytest
from src.embeddings import EmbeddingGenerator

def test_embedding_generation():
    # Initialize the embedding generator
    embedding_generator = EmbeddingGenerator(model_name="distilbert-base-uncased")

    # Sample markdown text
    markdown_text = "# Sample Title\n\nThis is a sample markdown text."

    # Generate embeddings
    embeddings = embedding_generator.generate_embeddings(markdown_text)

    # Check that embeddings are not None and have the expected shape
    assert embeddings is not None
    assert len(embeddings) > 0
    assert len(embeddings[0]) == embedding_generator.embedding_dimension

def test_embedding_generation_with_table():
    # Initialize the embedding generator
    embedding_generator = EmbeddingGenerator(model_name="distilbert-base-uncased")

    # Sample markdown text with a table
    markdown_text = """
    | Header 1 | Header 2 |
    |----------|----------|
    | Row 1    | Data 1   |
    | Row 2    | Data 2   |
    """

    # Generate embeddings
    embeddings = embedding_generator.generate_embeddings(markdown_text)

    # Check that embeddings are not None and have the expected shape
    assert embeddings is not None
    assert len(embeddings) > 0
    assert len(embeddings[0]) == embedding_generator.embedding_dimension

def test_invalid_model_name():
    # Initialize the embedding generator with an invalid model name
    with pytest.raises(ValueError):
        embedding_generator = EmbeddingGenerator(model_name="invalid-model-name")