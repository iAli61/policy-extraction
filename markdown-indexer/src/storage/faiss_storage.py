from typing import List, Tuple
import faiss
import numpy as np
from .base import BaseStorage

class FaissStorage(BaseStorage):
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # Using L2 distance
        self.documents = []

    def add_documents(self, embeddings: np.ndarray, documents: List[str]):
        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, query_embedding: np.ndarray, k: int) -> List[Tuple[int, str]]:
        distances, indices = self.index.search(query_embedding, k)
        results = [(distances[i][j], self.documents[indices[i][j]]) for i in range(len(query_embedding)) for j in range(k)]
        return results

    def get_document(self, index: int) -> str:
        return self.documents[index] if index < len(self.documents) else None

    def __len__(self):
        return len(self.documents)