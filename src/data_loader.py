import json

def load_restaurant_data(file_path: str) -> list:
    """Tải dữ liệu quán ăn từ file JSON."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def format_restaurant_for_embedding(restaurant: dict) -> str:
    """Định dạng thông tin quán ăn thành chuỗi văn bản để nhúng."""
    return f"{restaurant['name']} - {restaurant['cuisine']} - {restaurant['location']} - Giá: {restaurant['price_range']}"