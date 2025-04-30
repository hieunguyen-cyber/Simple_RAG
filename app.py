import streamlit as st
from src.chatbot import RestaurantChatbot
import wget
import os

# ID của file từ Google Drive và tên file cần tải
file_id = '1XOhWiIEpXccO5cTFXakt0tUINyQNnG7w'  # Lấy ID từ link của bạn
file_name = 'output.zip'   # Tên file cần tải về

# Tải file từ Google Drive bằng wget
url = f'https://docs.google.com/uc?export=download&id={file_id}'
wget.download(url, file_name)

# Tạo thư mục models nếu chưa có
os.makedirs('./models', exist_ok=True)

# Di chuyển file vào thư mục ./models
os.rename(file_name, f'./models/{file_name}')

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