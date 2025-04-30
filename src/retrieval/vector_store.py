# src/retrieval/vector_store.py

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import numpy as np
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
        self.collection = None
    
    def add_documents(self, documents: List[str], embeddings: List[np.ndarray], ids: List[str]):
        langchain_docs = [Document(page_content=doc, metadata={"id": id}) for doc, id in zip(documents, ids)]
        self.collection = Chroma.from_documents(
            documents=langchain_docs,
            embedding=self.embedding_function,
            ids=ids,
            persist_directory="./chroma_db"
        )
        self.collection.persist()
    
    def query(self, query_embedding: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        results = self.collection.similarity_search_by_vector(
            embedding=query_embedding,
            k=top_k
        )
        ids = [doc.metadata["id"] for doc in results]
        distances = [1 - np.dot(query_embedding, doc.vector) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc.vector)) 
                     if hasattr(doc, "vector") else 1.0 for doc in results]
        return {
            "ids": [ids],
            "distances": [distances]
        }