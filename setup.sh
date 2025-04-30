#!/bin/bash

# Cài đặt nếu chưa có huggingface_hub
pip install -r requirements.txt

# Thư mục lưu mô hình
MODEL_DIR="models"
MODEL_REPO="lmstudio-community/gemma-2-2b-it-GGUF"
MODEL_FILE="gemma-2-2b-it-Q4_K_M.gguf"

# Tạo thư mục nếu chưa có
mkdir -p $MODEL_DIR

echo "🔍 Đang tải mô hình từ Hugging Face ($MODEL_REPO)..."
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='$MODEL_REPO',
    filename='$MODEL_FILE',
    local_dir='$MODEL_DIR',
    local_dir_use_symlinks=False
)
"

echo "✅ Mô hình đã được lưu vào: $MODEL_DIR/$MODEL_FILE"