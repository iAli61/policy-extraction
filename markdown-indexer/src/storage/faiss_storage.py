from typing import List, Tuple
import faiss
import numpy as np
import pickle
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

    def add(self, data):
        embedding, document = data
        # Convert PyTorch tensor to numpy array correctly
        if hasattr(embedding, 'detach'):
            # This is a PyTorch tensor
            embedding_array = embedding.detach().cpu().numpy()
        else:
            # Already a numpy array or similar
            embedding_array = np.array(embedding)
        
        # Ensure the array has the right shape (1, dimension)
        if embedding_array.ndim == 1:
            embedding_array = embedding_array.reshape(1, -1)
            
        self.add_documents(embedding_array, [document])

    def retrieve(self, query):
        query_embedding = np.array([query])
        return self.search(query_embedding, k=1)

    def save(self, index_path: str, documents_path: str):
        # Save the FAISS index
        faiss.write_index(self.index, index_path)
        # Save the documents list
        with open(documents_path, 'wb') as f:
            pickle.dump(self.documents, f)