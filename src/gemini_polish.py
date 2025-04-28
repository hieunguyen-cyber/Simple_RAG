import streamlit as st
from google import genai

def polish_response(response: str) -> str:
    """Tinh chỉnh câu trả lời với Gemini."""
    # Cấu hình API key
    api_key=st.secrets["GOOGLE_API_KEY"]
    client = genai.Client(api_key=api_key)

    # Định nghĩa prompt
    prompt = f"""Hãy chỉnh sửa câu trả lời sau để ngắn gọn, tự nhiên và dễ hiểu hơn bằng tiếng Việt:

    {response}

    Trả lời:
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=[prompt]
        )
        result_text = response.text.strip()
        return result_text
    except Exception as e:
        print(f"Lỗi khi gọi API Gemini: {e}")
        return response