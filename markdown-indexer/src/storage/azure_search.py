class AzureSearchStorage(BaseStorage):
    def __init__(self, azure_search_client, index_name):
        self.client = azure_search_client
        self.index_name = index_name

    def add_documents(self, documents):
        actions = [
            {"@search.action": "upload", "id": doc["id"], "content": doc["content"]}
            for doc in documents
        ]
        self.client.index_documents(self.index_name, actions)

    def search(self, query):
        results = self.client.search(self.index_name, query)
        return results

    def index_markdown(self, markdown_text, embedding_model):
        processor = MarkdownProcessor()
        chunks = processor.chunk_markdown(markdown_text)

        embeddings = EmbeddingGenerator().generate_embeddings(chunks, embedding_model)
        documents = [{"id": str(i), "content": chunk, "embedding": embedding} for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))]

        self.add_documents(documents)