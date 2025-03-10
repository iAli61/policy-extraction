from src.markdown_processor import MarkdownProcessor
from src.embeddings import EmbeddingGenerator
from src.storage.faiss_storage import FaissStorage

def index_markdown_with_faiss(markdown_text, model_name):
    # Initialize the Markdown processor
    processor = MarkdownProcessor()
    
    # Parse and chunk the markdown text
    chunks = processor.chunk_markdown(markdown_text)
    
    # Initialize the embedding generator
    embedding_generator = EmbeddingGenerator(model_name=model_name)
    
    # Generate embeddings for each chunk
    embeddings = [embedding_generator.generate_embedding(chunk) for chunk in chunks]
    
    # Initialize FAISS storage
    faiss_storage = FaissStorage()
    
    # Index the embeddings
    for chunk, embedding in zip(chunks, embeddings):
        faiss_storage.add(embedding, chunk)
    
    print(f"Indexed {len(chunks)} chunks into FAISS.")

if __name__ == "__main__":
    # Example markdown text
    markdown_text = """
    # Sample Document

    This is a sample markdown document.

    | Header 1 | Header 2 |
    |----------|----------|
    | Row 1    | Data 1   |
    | Row 2    | Data 2   |

    More text follows here.
    """
    
    # Specify the Hugging Face model name
    model_name = "distilbert-base-uncased"

    # Index the markdown text
    index_markdown_with_faiss(markdown_text, model_name)