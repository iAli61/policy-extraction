class Indexer:
    def __init__(self, markdown_processor, storage_backend, embedding_generator):
        self.markdown_processor = markdown_processor
        self.storage_backend = storage_backend
        self.embedding_generator = embedding_generator

    def index_markdown(self, markdown_text):
        chunks = self.markdown_processor.process(markdown_text)
        for chunk in chunks:
            embedding = self.embedding_generator.generate_embedding(chunk)
            self.storage_backend.add(embedding, chunk)

    def search(self, query):
        query_embedding = self.embedding_generator.generate_embedding(query)
        return self.storage_backend.search(query_embedding)