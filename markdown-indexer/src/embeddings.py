from transformers import AutoModel, AutoTokenizer
import torch

class EmbeddingGenerator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings
    