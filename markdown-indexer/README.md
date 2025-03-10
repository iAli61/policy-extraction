# Markdown Indexer

This project provides a framework for indexing Markdown text using embedding models from Hugging Face. It supports two storage backends: FAISS and Azure AI Search. The indexing process is designed to handle tables effectively, ensuring that either the entire table is included in a chunk or, if too large, the header and one or more rows are included.

## Features

- **Markdown Processing**: Efficiently parses and chunks Markdown text, including tables.
- **Embedding Generation**: Utilizes Hugging Face models to generate embeddings for text.
- **Storage Backends**: Supports indexing and searching using FAISS and Azure AI Search.
- **Flexible Chunking**: Specializes in chunking text and tables to optimize indexing.

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

### Indexing with FAISS

To index Markdown text using FAISS, you can use the provided example script:

```python
from src.indexer import Indexer

indexer = Indexer(storage_backend='faiss')
indexer.index_markdown('path/to/your/markdown_file.md')
```

### Indexing with Azure AI Search

To index Markdown text using Azure AI Search, use the following example:

```python
from src.indexer import Indexer

indexer = Indexer(storage_backend='azure')
indexer.index_markdown('path/to/your/markdown_file.md')
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.