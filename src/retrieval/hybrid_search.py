# src/retrieval/hybrid_search.py

import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
import pandas as pd
from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import VectorStore

class HybridRetriever:
    def __init__(self, df: pd.DataFrame, vector_store: VectorStore, embedder: Embedder, alpha: float = 0.5):
        self.df = df
        self.vector_store = vector_store
        self.embedder = embedder
        self.alpha = alpha
        tokenized_corpus = [doc.lower().split() for doc in df['description']]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def retrieve(self, query: str, filtered_df: pd.DataFrame, top_k: int = 3) -> List[Dict[str, Any]]:
        filtered_indices = filtered_df.index.tolist()
        filtered_texts = filtered_df['description'].tolist()
        filtered_ids = [str(row['id']) for _, row in filtered_df.iterrows()]
        
        if not filtered_texts:
            return []
        
        query_embedding = self.embedder.embed([query])[0]
        dense_results = self.vector_store.query(query_embedding, top_k=top_k * 2)
        dense_ids = [id for id in dense_results['ids'][0] if id in filtered_ids]
        dense_scores = [1 - dist for dist, id in zip(dense_results['distances'][0], dense_results['ids'][0]) if id in filtered_ids]
        
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_scores_filtered = [bm25_scores[i] for i in filtered_indices]
        bm25_top_k = np.argsort(bm25_scores_filtered)[::-1][:top_k * 2]
        bm25_ids = [filtered_ids[i] for i in bm25_top_k]
        bm25_scores = [bm25_scores_filtered[i] for i in bm25_top_k]
        
        dense_scores = np.array(dense_scores) / np.max(dense_scores) if dense_scores else dense_scores
        bm25_scores = np.array(bm25_scores) / np.max(bm25_scores) if bm25_scores else bm25_scores
        
        combined_scores = {}
        for idx, dense_id in enumerate(dense_ids):
            combined_scores[int(dense_id)] = combined_scores.get(int(dense_id), 0) + self.alpha * dense_scores[idx]
        for idx, bm25_id in enumerate(bm25_ids):
            combined_scores[int(bm25_id)] = combined_scores.get(int(bm25_id), 0) + (1 - self.alpha) * bm25_scores[idx]
        
        sorted_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)[:top_k]
        return [self.df[self.df['id'] == id].iloc[0].to_dict() for id in sorted_ids]