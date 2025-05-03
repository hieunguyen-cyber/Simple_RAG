import gradio as gr
from src.chatbot import RestaurantChatbot

chatbot = RestaurantChatbot()
chat_history = []

def respond(user_message, history):
    response, retrieved_docs = chatbot.answer(user_message)

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

    return bot_response

with gr.Blocks() as demo:
    gr.Markdown("## Chatbot Gợi ý Quán ăn")
    chatbot_ui = gr.ChatInterface(fn=respond, chatbot=gr.Chatbot())

demo.launch()