from FlagEmbedding import BGEM3FlagModel
import numpy as np
from typing import List

class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model = BGEM3FlagModel(model_name, use_fp16=True)
    
    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=12,
            max_length=8192
        )['dense_vecs']
        return np.array(embeddings)