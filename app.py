import subprocess
import os

# Tải mô hình nếu chưa có (gọi shell script)
if not os.path.exists("models/gemma-2-2b-it-Q4_K_M.gguf"):
    subprocess.run(["bash", "setup.sh"], check=True)

# app.py
import torch
torch._C._get_custom_class_python_wrapper = lambda name, attr: None  # Bypass PyTorch custom class registration
from flask import Flask, request, render_template_string
from src.chatbot import RestaurantChatbot
import pickle

CACHE_FILE = "cache.pkl"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
app = Flask(__name__)
chatbot = RestaurantChatbot()
messages = []

@app.route("/", methods=["GET", "POST"])
def index():
    cache = load_cache()
    bot_response = ""
    if request.method == "POST":
        user_query = request.form.get("query")
        if user_query:
            messages.append({"role": "user", "content": user_query})
            if user_query in cache:
                bot_response = cache[user_query]
            else:
                response, retrieved_docs = chatbot.answer(user_query)
                bot_response = f"{response}\n\n**Nhà hàng gợi ý:**\n"
                if retrieved_docs:
                    for doc in retrieved_docs:
                        bot_response += (
                            f"- **{doc['name']} ({doc['cuisine']})**\n"
                            f"  - Món ăn: {', '.join(doc['dishes'])}\n"
                            f"  - Giá: {doc['price_range']}\n"
                            f"  - Khoảng cách: {doc['distance']} km\n"
                            f"  - Đánh giá: {doc['rating']}\n"
                            f"  - Địa chỉ: {doc['address']}\n"
                            f"  - Mô tả: {doc['description']}\n"
                        )
                else:
                    bot_response += "- Không tìm thấy nhà hàng phù hợp."
                cache[user_query] = bot_response
                save_cache(cache)
            messages.append({"role": "assistant", "content": bot_response})

    # HTML template
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chatbot Gợi ý Quán ăn</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .chat-container { max-width: 600px; margin: auto; }
            .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
            .user { background-color: #e6f3ff; }
            .assistant { background-color: #f0f0f0; }
            form { margin: 20px 0; }
            input[type="text"] { width: 80%; padding: 8px; }
            input[type="submit"] { padding: 8px 16px; }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <h1>Chatbot Gợi ý Quán ăn</h1>
            <form method="POST">
                <input type="text" name="query" placeholder="Nhập yêu cầu (ví dụ: Quán ăn Việt Nam gần đây, giá rẻ)" required>
                <input type="submit" value="Gửi">
            </form>
            <div>
                {% for message in messages %}
                    <div class="message {{ message.role }}">
                        <b>{{ message.role }}:</b> {{ message.content | replace('\n', '<br>') }}
                    </div>
                {% endfor %}
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html, messages=messages)

if __name__ == "__main__":
    app.run(debug=True)