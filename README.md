Project_Team_9/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                       # Hai file dữ liệu gốc
│   ├── simulated/                 # Dữ liệu động mô phỏng
│   └── processed/
│       └── data_khach_hang.csv     # Kết quả Stage 1, đầu vào Stage 2
│
├── notebooks/
│   ├── Stage1.ipynb
│   ├── PCA_KMeans.ipynb
│   ├── Regression.ipynb
│   └── Classification.ipynb
│
├── models/                        # Mô hình và bộ chuẩn hóa đã lưu
│
├── app/
│   └── app.py                     # Giao diện Streamlit hoặc Gradio
│
└── reports/
    ├── BaoCao.docx
    ├── Slides.pptx
    └── images/                    # Biểu đồ và ảnh chụp giao diện
