from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.prompts import PromptTemplate
import os
from typing import List

class LLM:
    def __init__(self, model_repo: str = "Qwen/Qwen2-1.5B-Instruct", 
                 local_path: str = "models"):
        """
        Initialize the LLM with Qwen2-1.5B-Instruct using Hugging Face Transformers.

        Args:
            model_repo (str): Hugging Face repository ID for the model.
            local_path (str): Local directory to store the model.
        """
        os.makedirs(local_path, exist_ok=True)
        
        try:
            # Load the model
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_repo,
                device_map="auto",  # Automatically map to CPU
                cache_dir=local_path,
                trust_remote_code=True
            )
            
            # Load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_repo,
                cache_dir=local_path,
                trust_remote_code=True
            )
            print(f"Model successfully loaded from {model_repo}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize model from {model_repo}. "
                f"Please ensure the model is available at https://huggingface.co/{model_repo}. "
                f"Error: {str(e)}"
            )
        
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
    
    def generate(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt (str): Input prompt.
            max_length (int): Maximum length of the generated text.
        
        Returns:
            str: Generated text.
        """
        try:
            # Apply chat template for instruction-tuned Qwen model
            messages = [{"role": "user", "content": prompt}]
            prompt_with_template = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Tokenize input prompt
            inputs = self.tokenizer(prompt_with_template, return_tensors="pt").to(self.llm.device)
            # Generate text
            outputs = self.llm.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
            # Decode the generated tokens
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("Response generated successfully!")
            return response.split('assistant')[2]
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {str(e)}")
    
    def format_query_prompt(self, query: str, cuisines: List[str], dishes: List[str], price_ranges: List[str]) -> str:
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
    # Khởi tạo đối tượng LLM với model_repo và local_path
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