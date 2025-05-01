# Sử dụng image base Python 3.12.9
FROM python:3.12.9-slim

RUN pip install --upgrade pip

# Thiết lập thư mục làm việc
WORKDIR /app

COPY requirements.txt .

# Cài đặt các thư viện Python
RUN pip install -r requirements.txt

# Copy mã nguồn vào container
COPY . .

# Mở cổng cần thiết
EXPOSE 8501

# Chạy ứng dụng streamlit
CMD ["streamlit", "run", "app.py"]