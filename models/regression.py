# -*- coding: utf-8 -*-
"""Pipeline Regression: dự đoán chi tiêu khách hàng trong 90 ngày."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42
TEST_SIZE = 0.20
FEATURES = [
    "RecencyDays", "Frequency", "MonetaryGBP", "TotalUnits",
    "AvgProductsPerInvoice", "ObservedTenureDays", "AvgOrderValueGBP"
]
REG_TARGET = "Chi_tieu_90_ngay_GBP"
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "Data/processed/data_regression.csv"
OUTPUT_DIR = ROOT / "models/ket_qua"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy dữ liệu: {DATA_PATH}")

data = pd.read_csv(DATA_PATH)
required = FEATURES + [REG_TARGET]
missing = [c for c in required if c not in data.columns]
if missing:
    raise ValueError(f"Thiếu cột bắt buộc: {missing}")
rows_before = len(data)
data = data.dropna(subset=required).reset_index(drop=True)
if data.empty:
    raise ValueError("Không còn bản ghi hợp lệ sau khi loại giá trị thiếu.")
if (data[required] < 0).any().any():
    raise ValueError("Các đặc trưng và chi tiêu được kỳ vọng không âm; hãy kiểm tra dữ liệu.")
print(f"Số bản ghi hợp lệ: {len(data)} (đã loại {rows_before-len(data)} dòng thiếu)")
print(f"Số đặc trưng: {len(FEATURES)}")
print(f"Tỷ lệ chi tiêu mục tiêu bằng 0: {(data[REG_TARGET] == 0).mean():.1%}")

X = data[FEATURES].copy()
y = data[REG_TARGET].copy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# Linear Regression: chỉ fit scaler trên tập train để tránh data leakage.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
linear_model = LinearRegression().fit(X_train_scaled, y_train)
pred_linear = linear_model.predict(X_test_scaled)

# Decision Tree không yêu cầu chuẩn hóa đặc trưng.
tree_model = DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)
tree_model.fit(X_train, y_train)
pred_tree = tree_model.predict(X_test)

predictions = {"Linear Regression": pred_linear, "Decision Tree Regressor": pred_tree}
rows = []
for name, pred in predictions.items():
    mse = mean_squared_error(y_test, pred)
    rows.append({
        "Mô hình": name,
        "MAE (GBP)": mean_absolute_error(y_test, pred),
        "MSE (GBP²)": mse,
        "RMSE (GBP)": math.sqrt(mse),
        "R²": r2_score(y_test, pred),
    })
results = pd.DataFrame(rows)
print("\nKẾT QUẢ SO SÁNH")
print(results.round(4).to_string(index=False))

# Chọn theo R² cao nhất; RMSE thấp hơn là tiêu chí phụ khi R² bằng nhau.
winner = results.sort_values(["R²", "RMSE (GBP)"], ascending=[False, True]).iloc[0]["Mô hình"]
best_pred = predictions[winner]
print(f"\nMô hình tốt hơn trong lần chia này: {winner}")
print(f"Số dự đoán âm của Linear Regression: {(pred_linear < 0).sum()}")

# Actual vs Predicted
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_test, best_pred, alpha=0.6, s=28)
lo = min(float(y_test.min()), float(np.min(best_pred)))
hi = max(float(y_test.max()), float(np.max(best_pred)))
ax.plot([lo, hi], [lo, hi], linestyle="--", label="Dự đoán = Thực tế")
ax.set_xlabel("Chi tiêu thực tế (GBP)")
ax.set_ylabel("Chi tiêu dự đoán (GBP)")
ax.set_title(f"Actual vs Predicted - {winner}")
ax.grid(alpha=0.2); ax.legend(); fig.tight_layout()
fig.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Residual Plot: residual = actual - predicted; dùng dự đoán thô để đánh giá.
residuals = y_test.to_numpy() - best_pred
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(best_pred, residuals, alpha=0.6, s=28)
ax.axhline(0, linestyle="--", label="Phần dư = 0")
ax.set_xlabel("Chi tiêu dự đoán (GBP)")
ax.set_ylabel("Phần dư: Thực tế - Dự đoán (GBP)")
ax.set_title(f"Residual Plot - {winner}")
ax.grid(alpha=0.2); ax.legend(); fig.tight_layout()
fig.savefig(OUTPUT_DIR / "residual_plot.png", dpi=200, bbox_inches="tight")
plt.close(fig)

results.to_csv(OUTPUT_DIR / "so_sanh_mo_hinh.csv", index=False, encoding="utf-8-sig")
out = pd.DataFrame({
    "Dong_du_lieu": X_test.index.to_numpy(),
    "Thuc_te_GBP": y_test.to_numpy(),
    "Du_doan_LinearRegression_GBP": pred_linear,
    "Du_doan_DecisionTree_GBP": pred_tree,
    "Phan_du_mo_hinh_duoc_chon_GBP": residuals,
})
if "CustomerID" in data.columns:
    out.insert(1, "CustomerID", data.loc[X_test.index, "CustomerID"].to_numpy())
out.to_csv(OUTPUT_DIR / "du_doan_tap_test.csv", index=False, encoding="utf-8-sig")
print(f"\nĐã lưu bảng kết quả và biểu đồ tại: {OUTPUT_DIR}")
