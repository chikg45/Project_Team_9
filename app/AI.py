# -*- coding: utf-8 -*-
# Chạy: streamlit run AI.py   (cần file data_khach_hang.csv nằm cùng thư mục)
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

st.set_page_config(page_title="Phân tích khách hàng", layout="wide")
st.title("📊 Phân tích khách hàng")

FEATURES = ["RecencyDays", "Frequency", "MonetaryGBP", "TotalUnits",
            "AvgProductsPerInvoice", "ObservedTenureDays", "AvgOrderValueGBP"]
LOG_COLS = ["Frequency", "MonetaryGBP", "TotalUnits", "AvgProductsPerInvoice", "AvgOrderValueGBP"]
TARGET = "Chi_tieu_90_ngay_GBP"
LABEL = "Mua_lai_90_ngay"
KM_FEATURES = ["Frequency", "MonetaryGBP", "TotalUnits", "AvgProductsPerInvoice",
               "RecencyDays", "ObservedTenureDays", "AvgOrderValueGBP"]  # đúng thứ tự file KMeans gốc
APP_DIR = Path(__file__).parent
KM_FILES = ["kmeans_pca.pkl", "comparison.csv", "scores_pca.csv", "scores_logscaled.csv",
            "cluster_profile.csv", "kmeans_elbow.png", "kmeans_silhouette.png", "kmeans_clusters.png"]


def km(name):
    """Tìm file kết quả KMeans trong thư mục ket_qua/ hoặc ngay cạnh AI.py."""
    for folder in (APP_DIR / "ket_qua", APP_DIR):
        if (folder / name).exists():
            return folder / name
    return None


def log_features(X):
    X = X.copy()
    X[LOG_COLS] = np.log1p(X[LOG_COLS])
    return X


@st.cache_data
def load_data():
    return pd.read_csv(Path(__file__).parent / "data_khach_hang.csv")


@st.cache_resource
def load_kmeans(df):
    # PCA 3 thành phần fit lại trên dữ liệu gốc cho ra đúng PC1-3 trong pca_features.csv;
    # KMeans (K=2) lấy từ mô hình đã lưu kmeans_pca.pkl
    X = log_features(df[KM_FEATURES])
    scaler = StandardScaler().fit(X)
    pca = PCA(n_components=3, random_state=42).fit(scaler.transform(X))
    return scaler, pca, joblib.load(km("kmeans_pca.pkl"))


@st.cache_resource
def train_regression(df):
    X_tr, X_te, y_tr, y_te = train_test_split(df[FEATURES], df[TARGET], test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_tr)
    lin = LinearRegression().fit(scaler.transform(X_tr), y_tr)
    tree = DecisionTreeRegressor(max_depth=5, random_state=42).fit(X_tr, y_tr)
    pred = {"LinearRegression": lin.predict(scaler.transform(X_te)),
            "DecisionTreeRegressor": tree.predict(X_te)}
    metrics = pd.DataFrame([{"Model": n, "RMSE (GBP)": np.sqrt(mean_squared_error(y_te, p)),
                             "R2": r2_score(y_te, p)} for n, p in pred.items()])
    return scaler, lin, tree, y_te, pred, metrics


@st.cache_resource
def train_classifier(df):
    # Giống Task 4: chia 80/20, stratify, Logistic (có scaler) và Decision Tree max_depth=3
    X_tr, X_te, y_tr, y_te = train_test_split(df[FEATURES], df[LABEL], test_size=0.2,
                                              stratify=df[LABEL], random_state=42)
    scaler = StandardScaler().fit(X_tr)
    logi = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    logi.fit(scaler.transform(X_tr), y_tr)
    tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42).fit(X_tr, y_tr)
    pred = {"Logistic Regression": logi.predict(scaler.transform(X_te)), "Decision Tree": tree.predict(X_te)}
    scores = pd.DataFrame({n: [accuracy_score(y_te, p), precision_score(y_te, p),
                               recall_score(y_te, p), f1_score(y_te, p)] for n, p in pred.items()},
                          index=["Accuracy", "Precision (mua lại)", "Recall (mua lại)", "F1 (mua lại)"])
    return scaler, logi, tree, y_te, pred, scores


df = load_data()

# ---------- Nhập số liệu một khách hàng (dùng cho cả 2 tab) ----------
st.sidebar.header("Nhập số liệu khách hàng")
inp = {c: st.sidebar.number_input(c, min_value=0.0, value=float(df[c].median())) for c in FEATURES}
new_row = pd.DataFrame([inp])

tab0, tab1, tab2, tab3 = st.tabs(["🏠 Tổng quan dữ liệu", "🧩 Phân cụm KMeans",
                                  "📈 Dự đoán chi tiêu 90 ngày", "🔁 Dự đoán mua lại (Task 4)"])

# ---------- Tab 0: Tổng quan dữ liệu ----------
with tab0:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Số khách hàng", f"{len(df):,}")
    m2.metric("Số quốc gia", df["Country"].nunique())
    m3.metric("Chi tiêu 90 ngày TB", f"{df[TARGET].mean():,.0f} GBP")
    m4.metric("Tỉ lệ mua lại 90 ngày", f"{df['Mua_lai_90_ngay'].mean():.1%}")

    st.subheader("Dữ liệu mẫu")
    st.dataframe(df.head(10))

    st.subheader("Thống kê các cột số")
    st.dataframe(df[FEATURES + [TARGET]].describe().T.round(2))

    c1, c2 = st.columns(2)
    c1.subheader("Top 10 quốc gia")
    c1.bar_chart(df["Country"].value_counts().head(10))
    c2.subheader("Kênh liên hệ")
    c2.bar_chart(df["Kenh_lien_he_mo_phong"].value_counts())

    c3, c4 = st.columns(2)
    col = c3.selectbox("Phân bố của", FEATURES + [TARGET])
    counts, edges = np.histogram(df[col], bins=20)
    c3.bar_chart(pd.Series(counts, index=[f"{e:,.0f}" for e in edges[:-1]]))
    c4.subheader(f"Tương quan với {TARGET}")
    c4.bar_chart(df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET))
    st.caption(f"Dữ liệu thiếu: {int(df.isna().sum().sum())} ô. "
               f"{(df[TARGET] == 0).mean():.0%} khách không chi tiêu gì trong 90 ngày.")

# ---------- Tab 1: KMeans (dùng kết quả đã làm sẵn) ----------
with tab1:
    thieu = [f for f in KM_FILES if km(f) is None]
    if thieu:
        st.error("Thiếu file KMeans: " + ", ".join(thieu) +
                 ". Hãy đặt chúng vào thư mục ket_qua/ nằm cạnh AI.py (hoặc ngay cạnh AI.py).")
    else:
        st.subheader("So sánh KMeans trên 7 đặc trưng (log + scale) và trên PCA")
        st.dataframe(pd.read_csv(km("comparison.csv")).round(4))

        c1, c2 = st.columns(2)
        c1.image(str(km("kmeans_elbow.png")))
        c2.image(str(km("kmeans_silhouette.png")))

        c3, c4 = st.columns(2)
        c3.write("Điểm theo K - PCA")
        c3.dataframe(pd.read_csv(km("scores_pca.csv")).round(4))
        c4.write("Điểm theo K - log + scale")
        c4.dataframe(pd.read_csv(km("scores_logscaled.csv")).round(4))

        st.subheader("Đặc điểm trung bình từng cụm (K = 2)")
        profile = pd.read_csv(km("cluster_profile.csv"), index_col=0, encoding="utf-8-sig")
        st.dataframe(profile)
        st.image(str(km("kmeans_clusters.png")))

        st.subheader("Khách hàng vừa nhập thuộc cụm nào?")
        if st.button("Phân cụm", key="btn_cluster"):
            scaler_k, pca_k, km = load_kmeans(df)
            z = pca_k.transform(scaler_k.transform(log_features(new_row[KM_FEATURES])))
            label = int(km.predict(z)[0])
            st.success(f"Khách hàng này thuộc **Cụm {label}**")
            st.write("Hồ sơ trung bình của cụm này:")
            st.dataframe(profile.loc[[label]])

# ---------- Tab 2: Regression ----------
with tab2:
     scaler_r, lin, tree_reg, y_te_r, pred_r, metrics_r = train_regression(df)
       
     st.subheader("So sánh 2 mô hình Regression (tập kiểm tra 20%)")
     st.dataframe(metrics_r.round(4))
   
     chosen_reg = st.selectbox("Xem biểu đồ thực tế vs dự đoán của mô hình", list(pred_r))
       
       # Vẽ biểu đồ scatter thực tế vs dự đoán
     chart_data = pd.DataFrame({
           "Thuc_te": y_te_r.to_numpy(), 
           "Du_doan": pred_r[chosen_reg]
       })
     st.scatter_chart(chart_data, x="Thuc_te", y="Du_doan")
   
     st.subheader("Dự đoán chi tiêu 90 ngày tới của khách hàng vừa nhập")
     if st.button("Dự đoán chi tiêu", key="btn_reg"):
           # Linear Regression dùng dữ liệu đã qua scaler_r
           p_lin = float(lin.predict(scaler_r.transform(new_row))[0])
           # Decision Tree Regressor dùng dữ liệu gốc mới nhập
           p_tree = float(tree_reg.predict(new_row)[0])
           
           a, b = st.columns(2)
           a.metric("Linear Regression", f"{max(p_lin, 0):,.2f} GBP")
           b.metric("Decision Tree", f"{max(p_tree, 0):,.2f} GBP")
           st.caption("Giá trị âm được đưa về 0.")

# ---------- Tab 3: Task 4 - Phân loại mua lại ----------
with tab3:
    scaler_c, logi, tree_c, y_te_c, pred_c, scores_c = train_classifier(df)
    st.subheader("So sánh 2 mô hình (tập kiểm tra 20%)")
    st.dataframe(scores_c.round(3))

    st.subheader("Confusion matrix (hàng = thực tế, cột = dự đoán)")
    cols = st.columns(2)
    for col, (name, p) in zip(cols, pred_c.items()):
        col.write(name)
        col.dataframe(pd.DataFrame(confusion_matrix(y_te_c, p),
                                   index=["Thực: Không mua lại", "Thực: Mua lại"],
                                   columns=["Đoán: Không", "Đoán: Mua lại"]))

    st.subheader("Khách vừa nhập có mua lại trong 90 ngày không?")
    if st.button("Dự đoán mua lại", key="btn_clf"):
        p_log = float(logi.predict_proba(scaler_c.transform(new_row))[0, 1])
        p_tree = float(tree_c.predict_proba(new_row)[0, 1])
        a, b = st.columns(2)
        a.metric("Logistic Regression", "Mua lại" if p_log >= 0.5 else "Không mua lại", f"xác suất {p_log:.0%}", delta_color="off")
        b.metric("Decision Tree", "Mua lại" if p_tree >= 0.5 else "Không mua lại", f"xác suất {p_tree:.0%}", delta_color="off")
