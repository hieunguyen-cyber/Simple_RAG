# Integral_Calculator(simple-version)

## Use a simple CNN for recognizing integral and use SciPy for calculation

## Structure
```Python
Integral_Calculator_Simple/
├── README.md                    # Tài liệu mô tả dự án
├── requirements.txt             # Danh sách thư viện cần thiết
├── deployment/                  # Triển khai ứng dụng
│   ├── app.py                   # Ứng dụng Streamlit chính
│   ├── onnx_inference.py        # Script để inference bằng ONNX
│   ├── model.onnx               # Mô hình đã chuyển đổi sang ONNX
│   └── configs/                 # Cấu hình ứng dụng
│       ├── app_config.yaml      # Config cho Streamlit
│       └── model_config.yaml    # Config cho mô hình
├── src/                         # Mã nguồn chính
│   ├── models/                  # Thư mục mô hình
│   │   ├── teacher_model.py     # Mô hình teacher (lớn hơn)
│   │   ├── student_model.py     # Mô hình student (nhỏ hơn qua distillation)
│   │   └── base_model.py        # Định nghĩa mô hình cơ bản
│   ├── distillation/            # Module giảm kích thước mô hình
│   │   ├── distill.py           # Logic huấn luyện distillation
│   │   ├── train_distillation.py# Script chạy distillation
│   └── quantization/            # Module lượng tử hóa mô hình
│       ├── quantize.py          # Script lượng tử hóa mô hình
│       ├── quantized_model.pth  # Mô hình đã lượng tử hóa
│       └── quantization_utils.py# Công cụ hỗ trợ lượng tử hóa
├── models/                      # Lưu các phiên bản mô hình
│   ├── original_model.pth       # Mô hình ban đầu
│   ├── student_model.pth        # Mô hình distillation (nhẹ hơn)
│   ├── quantized_model.pth      # Mô hình lượng tử hóa
├── utils/                       # Các tiện ích hỗ trợ
│   ├── preprocess.py            # Tiền xử lý hình ảnh đầu vào
│   ├── postprocess.py           # Hậu xử lý đầu ra OCR
│   ├── logger.py                # Hỗ trợ log thông tin
│   ├── metrics.py               # Các hàm đánh giá hiệu năng mô hình
├── datasets/                    # Bộ dữ liệu
│   ├── train/                   # Dữ liệu huấn luyện
│   ├── test/                    # Dữ liệu kiểm tra
│   ├── preprocess_dataset.py    # Script tiền xử lý bộ dữ liệu
├── export_onnx.py               # Script để chuyển đổi mô hình sang ONNX
├── tests/                       # Thư mục kiểm thử
│   ├── test_distillation.py     # Kiểm thử distillation
│   ├── test_quantization.py     # Kiểm thử lượng tử hóa
│   ├── test_onnx_inference.py   # Kiểm thử inference ONNX
├── scripts/                     # Các script hỗ trợ
│   ├── train_model.py           # Huấn luyện mô hình ban đầu
│   ├── evaluate_model.py        # Đánh giá mô hình
│   ├── visualize_results.py     # Hiển thị kết quả OCR
├── LICENSE                      # Thông tin bản quyền
└── .gitignore                   # File gitignore cho dự án