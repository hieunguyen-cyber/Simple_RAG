from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.data_loader import load_restaurant_data, format_restaurant_for_embedding
import os

def create_vector_store(data_path: str = "data/restaurants.json") -> Chroma:
    """Tạo hoặc tải Chroma vector store từ dữ liệu quán ăn."""
    persist_directory = "chroma_db"
    
    # Kiểm tra nếu vector store đã tồn tại
    if os.path.exists(persist_directory):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
        vector_store = Chroma(
            collection_name="restaurants",
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        return vector_store
    
    # Tạo vector store mới nếu chưa tồn tại
    restaurants = load_restaurant_data(data_path)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    texts = [format_restaurant_for_embedding(restaurant) for restaurant in restaurants]
    metadatas = [{"name": restaurant["name"], "index": i} for i, restaurant in enumerate(restaurants)]
    
    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name="restaurants",
        persist_directory=persist_directory
    )
    
    return vector_store