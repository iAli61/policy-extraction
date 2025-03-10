from src.markdown_processor import MarkdownProcessor
from src.embeddings import EmbeddingGenerator
from src.storage.azure_search import AzureSearchStorage

def index_markdown_with_azure(markdown_text, azure_search_service, index_name, hf_model_name):
    # Initialize the Markdown processor
    markdown_processor = MarkdownProcessor()
    
    # Parse and chunk the markdown text
    chunks = markdown_processor.process(markdown_text)
    
    # Initialize the embedding generator
    embedding_generator = EmbeddingGenerator(hf_model_name)
    
    # Initialize the Azure Search storage
    azure_storage = AzureSearchStorage(azure_search_service, index_name)
    
    # Index each chunk
    for chunk in chunks:
        # Generate embeddings for the chunk
        embeddings = embedding_generator.generate(chunk)
        
        # Index the chunk with its embeddings
        azure_storage.index(chunk, embeddings)
        print(f"Indexed chunk: {chunk[:30]}...")  # Print the first 30 characters of the chunk for reference

if __name__ == "__main__":
    # Example usage
    markdown_text = """
    # Sample Markdown

    | Header 1 | Header 2 |
    |----------|----------|
    | Row 1    | Data 1   |
    | Row 2    | Data 2   |
    """

    azure_search_service = "your_azure_search_service"
    index_name = "your_index_name"
    hf_model_name = "your_hugging_face_model"

    index_markdown_with_azure(markdown_text, azure_search_service, index_name, hf_model_name)