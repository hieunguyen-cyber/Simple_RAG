from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
import os
import torch
import requests



class LLM:
    def __init__(self, model_file: str = "gemma-2-2b-it-Q4_K_M.gguf", local_path: str = "models"):
        # Ensure local directory exists
        model_path = os.path.join(local_path, model_file)
        # Check if model exists, otherwise raise error
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. Please download it manually."
            )
        # Initialize LlamaCpp model
        try:
            self.llm = LlamaCpp(
                model_path=model_path,
                n_ctx=2048,
                max_tokens=300,
                temperature=0.7,
                n_gpu_layers=-1 if torch.cuda.is_available() or torch.backends.mps.is_available() else 0,
                n_threads=8,
                verbose=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LlamaCpp with model {model_path}: {str(e)}")
        
        # Define prompt template for query parsing (used in query_parser.py)
        self.prompt_template = PromptTemplate(
            template="""Bạn là một trợ lý phân tích truy vấn nhà hàng. Phân tích truy vấn sau và trích xuất các đặc trưng: cuisine, menu, price_range, distance, rating, và description. Chỉ trích xuất các giá trị khớp chính xác với danh sách giá trị hợp lệ. Nếu không tìm thấy giá trị khớp, trả về null (hoặc [] cho menu). Loại bỏ các từ khóa đã trích xuất khỏi description. Trả về kết quả dưới dạng JSON.

**Danh sách giá trị hợp lệ**:
- cuisine: {cuisines}
- menu: {dishes}
- price_range: {price_ranges}

**Hướng dẫn**:
- cuisine: Chỉ chọn giá trị từ danh sách cuisine. Ví dụ, "Viet" → "Vietnamese".
- menu: Chỉ chọn các món khớp chính xác với danh sách menu. Ví dụ, "phở bò" → "phở", "sushi" → [].
- price_range: Chỉ chọn {price_ranges}. Ví dụ, "cheap" → "low".
- distance: Trích xuất số km (e.g., "2 km" → 2.0) hoặc từ khóa ["nearby", "close" → 2.0, "far" → 10.0]. Nếu không rõ, trả về null.
- rating: Trích xuất số (e.g., "4 stars" → 4.0). Nếu không rõ, trả về null.
- description: Phần còn lại sau khi loại bỏ các từ khóa đã trích xuất. Nếu rỗng, trả về truy vấn gốc.

**Truy vấn**: {query}

**Định dạng đầu ra**:
{{
  "cuisine": null | "tên loại ẩm thực",
  "menu": [],
  "price_range": null | "low" | "medium" | "high",
  "distance": null | số km | "nearby" | "close" | "far",
  "rating": null | số,
  "description": "phần mô tả còn lại"
}}
""",
            input_variables=["cuisines", "dishes", "price_ranges", "query"]
        )
    
    def generate(self, prompt: str, max_length: int = 300) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt (str): Input prompt.
            max_length (int): Maximum length of the generated text (overridden by max_tokens).
        
        Returns:
            str: Generated text.
        """
        try:
            response = self.llm.invoke(prompt, max_tokens=max_length)
            print("Response generated sucessfully!")
            return response.strip()
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {str(e)}")
    
    def format_query_prompt(self, query: str, cuisines: list, dishes: list, price_ranges: list) -> str:
        """
        Format the prompt for query parsing using the prompt template.
        
        Args:
            query (str): User query.
            cuisines (list): List of valid cuisines.
            dishes (list): List of valid dishes.
            price_ranges (list): List of valid price ranges.
        
        Returns:
            str: Formatted prompt.
        """
        return self.prompt_template.format(
            cuisines=cuisines,
            dishes=dishes,
            price_ranges=price_ranges,
            query=query
        )
if __name__ == "__main__":
    # Khởi tạo đối tượng LLM với model_file và local_path
    model_file = 'gemma-2-2b-it-Q4_K_M.gguf'
    local_path = 'models'
    
    try:
        # Khởi tạo đối tượng LLM
        llm = LLM(model_file=model_file, local_path=local_path)

        # Định nghĩa một truy vấn và các tham số cần thiết
        query = "Tìm quán ăn Việt Nam gần đây, giá rẻ với món phở và cơm tấm"
        cuisines = ["Vietnamese", "Chinese", "Italian"]
        dishes = ["phở", "sushi", "pasta", "cơm tấm"]
        price_ranges = ["low", "medium", "high"]

        # Sử dụng hàm generate để tạo câu trả lời từ truy vấn
        generated_text = llm.generate(query, max_length=300)

        # In kết quả ra màn hình
        print("Generated text:")
        print(generated_text)

    except Exception as e:
        print(f"Error: {str(e)}")