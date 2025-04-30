# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.chatbot import RestaurantChatbot
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
chatbot = RestaurantChatbot()

# CORS để iOS app/FE gọi API không bị lỗi bảo mật
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # đổi lại khi deploy thật
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    response, docs = chatbot.answer(query.question)
    return {
        "response": response,
        "docs": docs
    }

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)