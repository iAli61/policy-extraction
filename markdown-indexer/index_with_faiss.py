from src.markdown_processor import MarkdownProcessor
from src.embeddings import EmbeddingGenerator
from src.storage.faiss_storage import FaissStorage
import os

def index_markdown_with_faiss(markdown_text, 
                              model_name, 
                              chunk_size= 4000, 
                              chunk_overlap=200,
                              max_table_size=2000,
                              output_dir=None):

    # check if output_dir is provided
    if output_dir:
        output_chunks = f"{output_dir}/chunks.jsonl"
        output_index = f"{output_dir}/faiss_index"
        output_documents = f"{output_dir}/documents.pkl"
    else:
        output_chunks = None
        output_index = None
        output_documents = None
    # Check if the model name is provided
    if not model_name:
        raise ValueError("Model name must be provided.")
    # Check if the markdown text is provided
    if not markdown_text:
        raise ValueError("Markdown text must be provided.")
    # Check if the output directory exists
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize the Markdown processor
    processor = MarkdownProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_table_size=max_table_size
    )
    
    # Parse and chunk the markdown text - using the correct method flow
    parsed_blocks = processor.parse_markdown(markdown_text)
    chunks = processor.chunk_text(parsed_blocks)

    # save chunks to a file if output_chunks is provided
    if output_chunks:
        processor.save_chunks(chunks, output_chunks)

    
    # Initialize the embedding generator
    embedding_generator = EmbeddingGenerator(model_name=model_name)
    
    # Generate embeddings for each chunk
    embeddings = [embedding_generator.generate_embeddings(chunk['content']) for chunk in chunks]
    
    # Initialize FAISS storage with the appropriate dimension
    dimension = embeddings[0].shape[1]  # Assuming all embeddings have the same dimension
    faiss_storage = FaissStorage(dimension=dimension)
    
    # Index the embeddings
    for chunk, embedding in zip(chunks, embeddings):
        faiss_storage.add((embedding, chunk['content']))

    # Save the FAISS index and documents
    if output_index and output_documents:
        faiss_storage.save(output_index, output_documents)
    
    print(f"Indexed {len(chunks)} chunks into FAISS.")

if __name__ == "__main__":
    
    # read the markdown text from a file
    with open("/home/azureuser/policy-extraction/markdown_files/test.md", "r") as f:
        markdown_text = f.read()
    
    # Specify the Hugging Face model name
    model_name = "distilbert-base-uncased"

    # Index the markdown text
    index_markdown_with_faiss(
        markdown_text,
        model_name,
        chunk_size=5000,
        chunk_overlap=200,
        max_table_size=2000,
        output_dir="output"
        )