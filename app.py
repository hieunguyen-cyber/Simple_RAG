import streamlit as st
from src.chatbot import RestaurantChatbot
import os
import requests


model_file = "gemma-2-2b-it-Q4_K_M.gguf"
local_path = "models"
model_path = os.path.join(local_path, model_file)
if not os.path.exists(model_path):
    url = "https://drive.usercontent.google.com/download?id=1XOhWiIEpXccO5cTFXakt0tUINyQNnG7w&export=download&authuser=0&confirm=t&uuid=c4470895-3d14-43b8-af6d-267f146abedb&at=APcmpowYmv7x8M0fJvmfp-djUnZb:1746033199358"
    output_file = "gemma-2-2b-it-Q4_K_M.gguf"

    headers = {
        "Host": "drive.usercontent.google.com",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Cookie": "SID=g.a000wQhn4l...; __Secure-1PSID=...; ..."  # Cookie rút gọn. Thay bằng bản đầy đủ nếu cần xác thực
    }

    response = requests.get(url, headers=headers, stream=True)

    with open(output_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Tải xong file: {output_file}")
    # Tạo thư mục models nếu chưa có
    os.makedirs('./models', exist_ok=True)

    # Di chuyển file vào thư mục ./models
    os.rename(model_file, f'./{local_path}/{model_file}')
    
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Chatbot Gợi ý Quán ăn")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Nhập yêu cầu của bạn (ví dụ: 'Quán ăn Việt Nam gần đây, giá rẻ'):")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    
    with st.spinner("Đang tìm gợi ý..."):
        chatbot = RestaurantChatbot()
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

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)