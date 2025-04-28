from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from src.vector_store import create_vector_store
from src.gemini_polish import polish_response
import time
import torch
from huggingface_hub import hf_hub_download

def create_rag_chain():
    """Tạo chuỗi RAG với Gemma-2-2B và ChromaDB, sử dụng LLaMA.cpp."""
    # Tạo vector store
    vector_store = create_vector_store()
    model_path = hf_hub_download(
        repo_id="lmstudio-community/gemma-2-2b-it-GGUF",
        filename="gemma-2-2b-it-Q4_K_M.gguf",
        local_dir="./models"
    )
    print(f"Mô hình đã được tải về: {model_path}")
    # Khởi tạo mô hình Gemma-2-2B với LLaMA.cpp
    llm = LlamaCpp(
        model_path="./models/gemma-2-2b-it-Q4_K_M.gguf",
        n_ctx=2048,
        max_tokens=200,
        temperature=0.7,
        n_gpu_layers=-1 if torch.backends.mps.is_available() else 0,
        n_threads=8,
        verbose=True
    )
    
    # Định nghĩa prompt
    prompt_template = """Dựa trên thông tin sau đây từ dữ liệu quán ăn:

    {context}

    Hãy trả lời câu hỏi sau của người dùng một cách chính xác và ngắn gọn, chỉ sử dụng thông tin từ dữ liệu cung cấp:

    Câu hỏi: {question}

    Trả lời:
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Tạo chuỗi RAG
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 1}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return rag_chain

def get_restaurant_recommendation(query: str) -> str:
    """Trả về gợi ý quán ăn dựa trên câu hỏi của người dùng."""
    start_time = time.time()
    rag_chain = create_rag_chain()
    chroma_time = time.time() - start_time
    print(f"Thời gian ChromaDB search: {chroma_time:.2f} giây")
    
    start_time = time.time()
    result = rag_chain({"query": query})
    generation_time = time.time() - start_time
    print(f"Thời gian generation: {generation_time:.2f} giây")
    
    polished_response = polish_response(result["result"])
    return polished_response