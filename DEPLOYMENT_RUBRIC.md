# Hướng dẫn triển khai và bằng chứng rubric — Project Team 9

## 1. Chạy bằng đúng một lệnh

Mở PowerShell ở **thư mục gốc repository** (nơi có `docker-compose.yml`):

```powershell
docker compose -f docker-compose.yml up --build
```

Truy cập Streamlit: http://localhost:8501

Không cần mở JupyterLab. Nếu cổng 8501 bị chiếm, dừng ứng dụng cũ trước khi chạy.

## 2. Kiến trúc multi-container

- `web`: build từ `app/Dockerfile`, chạy Streamlit ở cổng 8501; có đầy đủ model và asset từ `app/`.
- `database`: chạy Qdrant ở cổng nội bộ 6333; không mở cổng ra host.
- `project_network`: mạng bridge riêng, web kết nối Qdrant bằng DNS `http://database:6333`.
- `qdrant_data`: Docker named volume, mount tại `/qdrant/storage`; lưu kết quả dự đoán sau khi container restart.

Trong Web UI: nhập 7 đặc trưng bên trái → nhấn **Dự đoán cả 3 kết quả** → xem nhóm KMeans, chi tiêu 90 ngày (GBP), xác suất mua lại. Dự đoán được lưu vào Qdrant và hiển thị trong mục lịch sử. Mô hình và scaler vẫn dựa trên code gốc của nhóm; app tái huấn luyện hồi quy/phân loại từ CSV khi khởi động, không tải các joblib Stage 2 cho hai nhiệm vụ đó.

## 3. Các lệnh kiểm tra khi báo cáo

```powershell
docker compose -f docker-compose.yml ps
docker compose -f docker-compose.yml logs --tail=100 web
docker compose -f docker-compose.yml logs --tail=100 database
docker compose -f docker-compose.yml exec web python app/smoke_test.py
docker volume ls
```

Để chứng minh volume: tạo ít nhất một dự đoán → `docker compose -f docker-compose.yml down` → `docker compose -f docker-compose.yml up -d` → kiểm tra lịch sử còn tồn tại. **Không dùng `down -v`** vì lệnh đó xóa volume dữ liệu.

## 4. Ảnh cần chụp cho báo cáo

1. Docker Desktop: hai container `web` và `database` cùng Running.
2. Trang Streamlit: 7 ô nhập liệu và 3 kết quả từ một lần dự đoán.
3. Tab PCA/KMeans: elbow, silhouette, scatter cluster và bảng so sánh.
4. Tab Regression: bảng MAE, MSE, RMSE, R²; biểu đồ thực tế–dự đoán.
5. Tab Classification: Accuracy, Precision, Recall, F1, confusion matrix.
6. Mục lịch sử dự đoán đọc từ Qdrant; chụp trước/sau restart.
7. `docker compose ps`, mạng `project_network`, volume `qdrant_data`.

## 5. Phần báo cáo phải bổ sung dựa trên số liệu thật

- Nêu cách chọn số thành phần PCA dựa trên explained variance và kết quả thực nghiệm, không chỉ ghi con số.
- Nêu cách chọn K của KMeans dựa trên elbow, silhouette và ý nghĩa kinh doanh của cụm.
- Nêu so sánh định lượng Regression/Classification từ dữ liệu kiểm tra (không suy đoán điểm số).
- Đề xuất 2–3 hành động kinh doanh có căn cứ, ví dụ ưu tiên chăm sóc nhóm có khả năng mua lại cao; chăm sóc giữ chân nhóm Recency lớn; cá nhân hóa ưu đãi cho nhóm chi tiêu cao. Xác nhận từng khuyến nghị bằng profile cụm và metric thực tế.
- Lưu ý: dữ liệu Online Retail mang tính nghiên cứu, không khẳng định mô hình đã được kiểm định cho triển khai thực tế.

## 6. Lưu ý

- Lệnh `docker build -t project-team-9 .` ở root chỉ build image **chạy notebook tự động**, không phải multi-container app. Dùng Docker Compose cho rubric.
- Thay đổi trong notebook không tự động được cập nhật vào Web App. Cần cập nhật các asset `app/` tương ứng và build lại.
- Không cần `docker login` để chạy local nếu image public tải được.
- Nếu không có Qdrant, dự đoán có thể vẫn hiển thị nhưng sẽ báo lỗi lưu; không tính là đạt yêu cầu database cho đến khi sửa.
