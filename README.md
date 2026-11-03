# Project Team 9 — Online Retail / Machine Learning

## Cấu trúc thực tế (giữ nguyên đường dẫn để notebook và app không bị hỏng)

```text
Project_Team_9/
├── Project_Team_9/
│   └── Project_Team_9_Tong_Stage1_Stage2.ipynb   # notebook tổng, chạy từ đầu đến cuối
├── notebooks/                                     # notebook gốc theo từng chủ đề
│   ├── Data_EDA.ipynb
│   ├── PCA.ipynb
│   ├── KMeans.ipynb
│   ├── Regression.ipynb
│   ├── Classification.ipynb
│   └── ket_qua/                                  # kết quả chạy và dự đoán (tạo tự động)
├── data_goc/
│   └── raw_data.zip                              # dữ liệu nguồn của Stage 1
├── Data/
│   ├── processed/                                # CSV dùng cho Stage 2
│   ├── simulated/
│   ├── csdl/                                     # SQLite và Qdrant local
│   └── data_goc/                                 # CSV giải nén khi chạy
├── models/                                       # mô hình và các script riêng
├── app/
│   ├── AI.py                                     # Streamlit dashboard
│   └── ...                                       # đầu ra KMeans, dữ liệu app, biểu đồ
├── report/                                       # báo cáo
├── run_notebook.py                               # chạy notebook tổng không cần bấm Run All
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

**Chú ý:** Không đổi tên `Data` thành `data` trên Linux/Docker. Hai thư mục `Data/data_goc` và `data_goc` khác nhau: ZIP gốc ở thư mục ngoài; khi chạy notebook sẽ giải nén vào `Data/data_goc`. Những file CSV và model được notebook tạo lại có thể ghi đè kết quả đã lưu.

## Chuẩn bị môi trường

Khuyến nghị Python 3.12. Ở thư mục gốc repository:

```powershell
python -m pip install -r requirements.txt
```

## Chạy trọn Stage 1 → Stage 2

```powershell
python run_notebook.py
```

Script đọc nguyên notebook tổng và thực thi lần lượt các code cell bằng một kernel mới, với thư mục gốc project làm điểm xuất phát. Khi kết thúc, kết quả nằm ở `notebooks/ket_qua/Project_Team_9_Tong_Stage1_Stage2_executed.ipynb`. Nếu gặp lỗi, script cũng lưu phần kết quả đã chạy được để điều tra traceback. **Không chỉnh sửa/ghi đè notebook gốc.**

Hoặc mở `Project_Team_9/Project_Team_9_Tong_Stage1_Stage2.ipynb` bằng JupyterLab từ thư mục gốc và chọn **Restart Kernel → Run All**. Notebook tự chuyển thư mục làm việc sang `Data/` để đọc CSV đầu vào.

Thứ tự: **Data_EDA → PCA → KMeans → Regression → Classification → kiểm tra app và kết luận**.

## Chạy Docker

```powershell
docker build -t project-team-9:latest .
docker run --rm -it --mount "type=bind,source=${PWD},target=/workspace" project-team-9:latest
```

Docker container mặc định thực thi notebook tổng. Lệnh trên dùng bind mount để lưu CSV, model và notebook kết quả về thư mục trên máy tính. Lần đầu có thể mất thời gian do tải/cài môi trường; Qdrant local và dữ liệu thô cần bộ nhớ/đĩa phù hợp.

## Chạy giao diện Streamlit

Sau khi chạy notebook (hoặc dùng kết quả có sẵn):

```powershell
python -m streamlit run app/AI.py
```

## Kiểm thử

GitHub Actions có workflow **Validate merged Stage 1 and Stage 2**, có thể chạy thủ công từ tab Actions. Workflow dùng Python 3.12 và lưu execution report dưới dạng artifact. Một lần CI thành công mới là bằng chứng chạy hết notebook trên GitHub; README này không tự khẳng định kiểm thử thành công.

## Nhật ký thay đổi

- Giữ nguyên notebook tổng, dữ liệu nguồn, notebook từng thành viên và các thuật toán.
- Chỉnh tài liệu cấu trúc thư mục để khớp repo thật.
- Thêm `run_notebook.py` để chạy đầy đủ 81 code cell theo thứ tự và lưu traceback/output.
- Thêm `Dockerfile`, `.dockerignore` và thư viện `nbclient`/`nbformat`.
- Thêm GitHub Actions để chạy notebook theo yêu cầu và tải artifact.
