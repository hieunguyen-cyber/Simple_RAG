from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
import os
from huggingface_hub import hf_hub_download

class LLM:
    def __init__(self, model_repo: str = "lmstudio-community/gemma-2-2b-it-GGUF", 
                 model_file: str = "gemma-2-2b-it-Q4_K_M.gguf", 
                 local_path: str = "models"):
        os.makedirs(local_path, exist_ok=True)
        model_path = os.path.join(local_path, model_file)
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Downloading {model_file} from {model_repo}...")
            try:
                model_path = hf_hub_download(
                    repo_id=model_repo,
                    filename=model_file,
                    local_dir=local_path,
                    local_dir_use_symlinks=False
                )
                print(f"Model downloaded and saved to {model_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download {model_file}. Please download manually from "
                    f"https://huggingface.co/{model_repo}/tree/main and place {model_file} in {local_path}. "
                    f"Error: {str(e)}"
                )
        try:
            self.llm = LlamaCpp(
                model_path=model_path,
                n_ctx=1024,  # Short context for low RAM
                max_tokens=200,  # Limit output length
                temperature=0.7,
                n_gpu_layers=0,  # No GPU on Vercel
                n_threads=2,  # Optimize for serverless
                verbose=False  # Reduce logging
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
    
    def generate(self, prompt: str, max_length: int = 1000) -> str:
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
    local_path = 'models'
    
    try:
        # Khởi tạo đối tượng LLM
        llm = LLM(local_path=local_path)

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