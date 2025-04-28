import streamlit as st
from src.rag_chain import get_restaurant_recommendation

# Khởi tạo lịch sử hội thoại
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tiêu đề ứng dụng
st.title("Chatbot Gợi ý Quán ăn")

# Hiển thị lịch sử hội thoại
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Ô nhập liệu kiểu chat
user_query = st.chat_input("Nhập yêu cầu của bạn (ví dụ: 'Quán ăn Việt Nam gần đây, giá rẻ'):")

# Xử lý khi người dùng gửi tin nhắn
if user_query:
    # Thêm tin nhắn của người dùng vào lịch sử
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Hiển thị tin nhắn của người dùng
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Lấy gợi ý từ RAG chain
    with st.spinner("Đang tìm gợi ý..."):
        recommendation = get_restaurant_recommendation(user_query)
    
    # Thêm tin nhắn của bot vào lịch sử
    st.session_state.messages.append({"role": "assistant", "content": recommendation})
    
    # Hiển thị tin nhắn của bot
    with st.chat_message("assistant"):
        st.markdown(recommendation)