import pandas as pd
from typing import Tuple, List, Dict, Any
from src.utils.data_loader import load_restaurant_data
from src.utils.query_parser import QueryParser
from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.keyword_filter import filter_restaurants
from src.retrieval.hybrid_search import HybridRetriever
from src.generation.llm import LLM
from langchain_core.embeddings import Embeddings

class LangChainEmbeddingWrapper(Embeddings):
    def __init__(self, embedder):
        self.embedder = embedder
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.embed(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.embedder.embed([text])[0].tolist()

class RestaurantChatbot:
    def __init__(self, data_path: str = "data/restaurants.json"):
        """
        Initialize the restaurant chatbot.
        
        Args:
            data_path (str): Path to the restaurant JSON file.
        """
        self.df = load_restaurant_data(data_path)
        self.embedder = Embedder()
        self.embedding_wrapper = LangChainEmbeddingWrapper(self.embedder)
        self.vector_store = VectorStore(embedding_function=self.embedding_wrapper)
        self.llm = LLM()
        self.parser = QueryParser(self.df)
        
        embeddings = self.embedder.embed(self.df['text'].tolist())
        self.vector_store.add_documents(
            documents=self.df['text'].tolist(),
            embeddings=embeddings.tolist(),
            ids=[str(i) for i in self.df['id']]
        )
        
        self.retriever = HybridRetriever(self.df, self.vector_store, self.embedder)
    
    def answer(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process a user query and return a natural, concise response with recommended restaurants.
        
        Args:
            query (str): User query.
        
        Returns:
            Tuple[str, List[Dict[str, Any]]]: Natural response text and list of recommended restaurants.
        """
        parsed_query = self.parser.parse_query(query)
        filtered_df = filter_restaurants(self.df, parsed_query)
        description = parsed_query["description"] if parsed_query["description"] else query
        
        if filtered_df.empty:
            retrieved_docs = self.retriever.retrieve(description, self.df, top_k=3)
        else:
            retrieved_docs = self.retriever.retrieve(description, filtered_df, top_k=3)
        
        if not retrieved_docs:
            return "Mình không tìm được nhà hàng nào phù hợp. Bạn thử đổi tiêu chí xem, như mở rộng khoảng cách hoặc loại món ăn nhé!", []
        
        # Create context for LLM
        context = "\n".join([
            f"- {doc['name']} ({doc['cuisine']}): {', '.join(doc['dishes'])}. "
            f"Price: {doc['price_range']}, Distance: {doc['distance']} km, Rating: {doc['rating']}. "
            f"Description: {doc['description']}"
            for doc in retrieved_docs
        ])
        
        # Prompt for natural, consultant-like response
        prompt = (
            f"Bạn là một người tư vấn nhà hàng thân thiện. Dựa trên truy vấn và danh sách nhà hàng, hãy gợi ý ngắn gọn, tự nhiên, như trò chuyện với bạn bè, giải thích tại sao chọn các nhà hàng này (tập trung vào món ăn, giá, khoảng cách, hoặc đánh giá phù hợp với truy vấn). Không lặp lại truy vấn hoặc dùng ngôn ngữ kỹ thuật. Chỉ dùng thông tin từ danh sách nhà hàng.\n\n"
            f"Truy vấn: {query}\n\n"
            f"Danh sách nhà hàng:\n{context}\n\n"
            f"Phản hồi:"
        )
        
        response = self.llm.generate(prompt, max_length=200)
        return response, retrieved_docs