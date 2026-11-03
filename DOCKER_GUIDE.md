# Hướng dẫn sử dụng Docker — Project Team 9

Tài liệu hướng dẫn chạy **JupyterLab** hoặc **Web UI Streamlit + Qdrant** từ cùng một repository.

## 1. Yêu cầu

- Cài [Docker Desktop](https://www.docker.com/products/docker-desktop/) và mở ứng dụng (Docker Engine đang chạy).
- Cài Git nếu cần tải/cập nhật code.
- Mở **PowerShell tại thư mục gốc dự án**, nơi có `Dockerfile`, `compose.yaml` và `docker-compose.yml`.

> **Lưu ý về branch:** Các lệnh dưới đây áp dụng cho phiên bản có cấu hình `jupyter` profile trong Docker Compose. Tại thời điểm viết hướng dẫn, cấu hình này nằm ở branch `fix/notebook-stage1-stage2-docker`. Sau khi merge branch đó vào `main`, chỉ cần dùng `main`.

### Lần đầu tải dự án

```powershell
cd C:\DockerProjects
git clone https://github.com/chikg45/Project_Team_9.git
cd Project_Team_9
```

Nếu **chưa merge cấu hình Compose vào `main`**, chuyển sang branch có cấu hình:

```powershell
git fetch origin
git switch fix/notebook-stage1-stage2-docker
```

Nếu **đã merge vào `main`**, dùng:

```powershell
git switch main
git pull origin main
```

Kiểm tra Docker Compose:

```powershell
docker compose config --services
```

Kết quả mặc định có `web` và `database`. Service `jupyter` nằm trong profile tùy chọn `notebook`.

## 2. Chế độ A — Chạy riêng JupyterLab

Trong PowerShell tại thư mục dự án:

```powershell
docker compose --profile notebook up --build jupyter
```

Mở **http://localhost:8888**. Nếu yêu cầu token, lấy đường dẫn/token được in trong PowerShell.

- Docker build image `project-team-9:latest` từ `Dockerfile` ở thư mục gốc.
- Container chạy JupyterLab với Python và các thư viện của dự án.
- Thư mục dự án được mount vào `/workspace`; file lưu trong thư mục này sẽ xuất hiện trên máy Windows.
- Notebook tổng: `Project_Team_9/Project_Team_9_Tong_Stage1_Stage2.ipynb`.
- Trong JupyterLab, chọn **Run → Run All Cells** nếu muốn chạy toàn bộ notebook.

**Dừng:** Nhấn `Ctrl + C`. Nếu muốn xóa container đã dừng, chạy `docker compose --profile notebook down` (không xóa dữ liệu notebook trên Windows).

## 3. Chế độ B — Chạy Web UI + Database theo rubric

Trong PowerShell tại thư mục dự án:

```powershell
docker compose up --build
```

Mở **http://localhost:8501**.

Compose chạy **hai service mặc định**:

| Service | Vai trò | Kết nối |
|---|---|---|
| `web` | Streamlit UI, Regression, KMeans, Classification | Trình duyệt: `localhost:8501` |
| `database` | Qdrant lưu vector và lịch sử dự đoán | Nội bộ Docker: `http://database:6333` |

Cả hai dùng Docker Network `project_network`. Qdrant sử dụng named volume `qdrant_data` để giữ dữ liệu qua các lần khởi động lại.

**Trên UI:** Nhập thông tin khách hàng → nhấn **Dự đoán cả 3 kết quả** → xem cụm KMeans, chi tiêu 90 ngày và khả năng mua lại. Phần lịch sử hiển thị các kết quả đã lưu vào Qdrant.

> Web App sử dụng dữ liệu/mô hình có trong `app/`; không bắt buộc phải chạy JupyterLab trước. Nếu bạn thay đổi dữ liệu hoặc model, hãy kiểm tra lại các asset mà Web App sử dụng.

### Kiểm tra trạng thái và chức năng

Mở **PowerShell thứ hai** tại cùng thư mục:

```powershell
docker compose ps
```

Kiểm thử giao diện, ba dự đoán và lưu dữ liệu Qdrant:

```powershell
docker compose exec web python app/smoke_test.py
```

Nếu có lỗi, xem log:

```powershell
docker compose logs --tail=100 web
docker compose logs --tail=100 database
```

**Dừng:** Nhấn `Ctrl + C`, sau đó có thể chạy:

```powershell
docker compose down
```

Lệnh `down` **không xóa** named volume. **Không dùng `docker compose down -v`** nếu muốn giữ lịch sử dự đoán.

## 4. Các lệnh cần nhớ

| Mục đích | Lệnh |
|---|---|
| Chạy riêng Jupyter | `docker compose --profile notebook up --build jupyter` |
| Chạy Web UI + Qdrant | `docker compose up --build` |
| Xem container | `docker compose ps` |
| Kiểm thử Web UI + Qdrant | `docker compose exec web python app/smoke_test.py` |
| Dừng và xóa container Compose | `docker compose down` |
| Xem log | `docker compose logs --tail=100` |

**Ghi nhớ:** `Image` là bản đóng gói; `Container` là ứng dụng được chạy từ image; `Docker Compose` cấu hình và khởi động nhiều service. `--build` yêu cầu build lại image của service có `build:`.

## 5. Lỗi thường gặp

- **`no configuration file provided`**: Chưa đứng ở thư mục gốc chứa `compose.yaml`/`docker-compose.yml`.
- **`port is already allocated`**: Cổng `8888` hoặc `8501` đang được ứng dụng/container khác sử dụng; dừng tiến trình cũ.
- **`Cannot connect to the Docker daemon`**: Mở Docker Desktop và đợi Docker Engine chạy.
- **Web mở nhưng không lưu được lịch sử**: Kiểm tra `docker compose ps` và `docker compose logs database`.
- **Jupyter yêu cầu password/token**: Copy token từ cửa sổ PowerShell đang chạy Jupyter.
- **Thay đổi code nhưng không thấy cập nhật trên Web**: Chạy lại `docker compose up --build`; service `web` copy thư mục `app/` vào image khi build.

## 6. Gợi ý trình bày với giảng viên

1. Chạy `docker compose up --build` để chứng minh **khởi động bằng một lệnh**.
2. Chạy `docker compose ps` để chứng minh **hai container Web + Database**.
3. Mở `localhost:8501`, nhập dữ liệu và trình bày **ba kết quả dự đoán**.
4. Mở lịch sử Qdrant; dừng rồi chạy lại Compose để minh họa **dữ liệu vẫn được giữ trong volume**.
5. Giải thích `project_network` kết nối `web` với `database` bằng hostname `database`.

> Chỉ kết luận hệ thống hoạt động đầy đủ sau khi chạy kiểm thử thành công trên máy trình bày.
