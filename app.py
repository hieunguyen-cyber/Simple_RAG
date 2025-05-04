import streamlit as st
from src.chatbot import RestaurantChatbot

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
        doc = retrieved_docs[0]
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