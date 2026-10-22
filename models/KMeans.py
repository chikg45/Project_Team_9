# -*- coding: utf-8 -*-
"""Chạy riêng phần phân tích; xem kết quả và biểu đồ trong thư mục ket_qua."""
import matplotlib
matplotlib.use("Agg")

def display(value):
    print(value.to_string() if hasattr(value, "to_string") else value)

# Bước 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / "Data/processed"
RESULT_DIR = ROOT / "app"
RESULT_DIR.mkdir(exist_ok=True)
FEATURES = [
    "RecencyDays", "Frequency", "MonetaryGBP", "TotalUnits",
    "AvgProductsPerInvoice", "ObservedTenureDays", "AvgOrderValueGBP"
]
PCA_FEATURES = [
    "Frequency", "MonetaryGBP", "TotalUnits", "AvgProductsPerInvoice",
    "RecencyDays", "ObservedTenureDays", "AvgOrderValueGBP"
]
LOG_COLUMNS = [
    "Frequency", "MonetaryGBP", "TotalUnits",
    "AvgProductsPerInvoice", "AvgOrderValueGBP"
]
# Biến tương lai chỉ dùng mô tả sau khi có cụm; không tham gia fit K-Means.
REG_TARGET = "Chi_tieu_90_ngay_GBP"
PC_COLUMNS = ["PC1", "PC2", "PC3"]

data = pd.read_csv(BASE_DIR / "data_khach_hang.csv")
pca_data = pd.read_csv(BASE_DIR / "pca_features.csv")
assert data["CustomerID"].is_unique and pca_data["CustomerID"].is_unique
assert set(data["CustomerID"]) == set(pca_data["CustomerID"]), "Hai tệp phải có cùng khách hàng."
# loc sắp xếp lại PC theo đúng thứ tự khách hàng của CSV gốc.
pca_aligned = pca_data.set_index("CustomerID").loc[data["CustomerID"], PC_COLUMNS]
X_pca = pca_aligned.to_numpy()
assert np.isfinite(X_pca).all(), "Dữ liệu PC thiếu hoặc không hợp lệ."
assert np.isfinite(data[FEATURES].to_numpy(dtype=float)).all()
assert (data[FEATURES] >= 0).all().all()
print("Số khách hàng:", len(data))
print("Kích thước dữ liệu PCA:", X_pca.shape)
display(data.head())
display(pca_aligned.head())

# Bước 2
X_log = data[PCA_FEATURES].copy()
X_log[LOG_COLUMNS] = np.log1p(X_log[LOG_COLUMNS])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)
print("Dữ liệu trước khi giảm chiều:", X_scaled.shape)
print("Dữ liệu sau PCA:", X_pca.shape)

# Bước 3
K_RANGE = range(2, 11)
ket_qua_cum_goc = []
ket_qua_cum_pca = []

for k in K_RANGE:
    # K-Means trên 7 đặc trưng đã log1p và chuẩn hóa
    model_goc = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    nhan_goc = model_goc.fit_predict(X_scaled)
    ket_qua_cum_goc.append({
        "K": k,
        "WCSS": model_goc.inertia_,
        "Silhouette": silhouette_score(X_scaled, nhan_goc)
    })

    # K-Means trên dữ liệu PCA
    model_pca = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    nhan_pca = model_pca.fit_predict(X_pca)
    ket_qua_cum_pca.append({
        "K": k,
        "WCSS": model_pca.inertia_,
        "Silhouette": silhouette_score(X_pca, nhan_pca)
    })

scores_goc = pd.DataFrame(ket_qua_cum_goc)
scores_pca = pd.DataFrame(ket_qua_cum_pca)

print("=== K-Means trên 7 đặc trưng log + scale ===")
display(scores_goc.round(4))
print("=== K-Means trên dữ liệu PCA ===")
display(scores_pca.round(4))

# Bước 4
# Elbow: WCSS theo K
plt.figure(figsize=(8, 5))
plt.plot(scores_goc["K"], scores_goc["WCSS"], marker="o", label="7 đặc trưng log + scale")
plt.plot(scores_pca["K"], scores_pca["WCSS"], marker="s", label="Dữ liệu PCA")
plt.xlabel("K")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(RESULT_DIR / "kmeans_elbow.png", dpi=150)
plt.close()

# Silhouette so sánh 7 đặc trưng log + scale và PCA
plt.figure(figsize=(8, 5))
plt.plot(scores_goc["K"], scores_goc["Silhouette"], marker="o", label="7 đặc trưng log + scale")
plt.plot(scores_pca["K"], scores_pca["Silhouette"], marker="s", label="Dữ liệu PCA")
plt.xlabel("K")
plt.ylabel("Silhouette score")
plt.title("Silhouette Analysis")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(RESULT_DIR / "kmeans_silhouette.png", dpi=150)
plt.close()

# Bước 5
# Chọn K có Silhouette cao nhất ở từng cách biểu diễn dữ liệu
best_k_goc = int(scores_goc.loc[scores_goc["Silhouette"].idxmax(), "K"])
best_k_pca = int(scores_pca.loc[scores_pca["Silhouette"].idxmax(), "K"])

print("K tốt nhất - 7 đặc trưng log + scale:", best_k_goc)
print("K tốt nhất - dữ liệu PCA:", best_k_pca)

# Dùng K của PCA để tạo cột Cluster_Label cho dữ liệu chính
K = best_k_pca
kmeans_goc = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=20)
labels_goc = kmeans_goc.fit_predict(X_scaled)
kmeans_pca = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=20)
labels_pca = kmeans_pca.fit_predict(X_pca)

comparison_cluster = pd.DataFrame({
    "Model": ["K-Means LogScaled", "K-Means PCA"],
    "K": [K, K],
    "WCSS": [kmeans_goc.inertia_, kmeans_pca.inertia_],
    "Silhouette": [
        silhouette_score(X_scaled, labels_goc),
        silhouette_score(X_pca, labels_pca)
    ]
})
display(comparison_cluster.round(4))
print("Lưu ý: không so sánh trực tiếp WCSS giữa hai không gian có số chiều khác nhau; Silhouette cũng phụ thuộc không gian, cần xem thêm kích thước và hồ sơ cụm.")

# Bước 6
# Gắn nhãn cụm vào dữ liệu và xem đặc điểm trung bình từng cụm
data_stage2 = data.copy()
data_stage2["Cluster_Label"] = labels_pca

profile_cluster = data_stage2.groupby("Cluster_Label")[FEATURES + [REG_TARGET]].mean().round(2)
profile_cluster.insert(0, "So_khach", data_stage2.groupby("Cluster_Label").size())
display(profile_cluster)

plt.figure(figsize=(8, 5))
for cluster_id in sorted(data_stage2["Cluster_Label"].unique()):
    mask = data_stage2["Cluster_Label"] == cluster_id
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=22, alpha=0.6, label=f"Cluster {cluster_id}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"K-Means trên PCA (K={K})")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(RESULT_DIR / "kmeans_clusters.png", dpi=150)
plt.close()

# Bước 7
data_stage2.to_csv(RESULT_DIR / "data_khach_hang_kmeans.csv", index=False, encoding="utf-8-sig")
scores_goc.to_csv(RESULT_DIR / "scores_logscaled.csv", index=False)
scores_pca.to_csv(RESULT_DIR / "scores_pca.csv", index=False)
comparison_cluster.to_csv(RESULT_DIR / "comparison.csv", index=False)
profile_cluster.to_csv(RESULT_DIR / "cluster_profile.csv", encoding="utf-8-sig")
joblib.dump(kmeans_pca, RESULT_DIR / "kmeans_pca.pkl")
print("Đã lưu code chạy riêng, biểu đồ, bảng kết quả và nhãn cụm tại:", RESULT_DIR)
