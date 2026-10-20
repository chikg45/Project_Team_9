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
KM_FEATURES = ["Frequency", "MonetaryGBP", "TotalUnits", "AvgProductsPerInvoice",
               "RecencyDays", "ObservedTenureDays", "AvgOrderValueGBP"]  # giữ đúng thứ tự lúc train
LOG_COLS = ["Frequency", "MonetaryGBP", "TotalUnits", "AvgProductsPerInvoice", "AvgOrderValueGBP"]
TARGET, LABEL = "Chi_tieu_90_ngay_GBP", "Mua_lai_90_ngay"
APP_DIR = Path(__file__).parent
KM_FILES = ["kmeans_pca.pkl", "comparison.csv", "scores_pca.csv", "scores_logscaled.csv",
            "cluster_profile.csv", "kmeans_elbow.png", "kmeans_silhouette.png", "kmeans_clusters.png"]


def km_file(name):
    """Tìm file kết quả KMeans trong thư mục ket_qua/ hoặc ngay cạnh file này."""
    return next((f for d in (APP_DIR / "ket_qua", APP_DIR) if (f := d / name).exists()), None)


def log_features(X):
    X = X.copy()
    X[LOG_COLS] = np.log1p(X[LOG_COLS])
    return X


@st.cache_data
def load_data():
    return pd.read_csv(APP_DIR / "data_khach_hang.csv")


@st.cache_resource
def load_kmeans(df):
    X = log_features(df[KM_FEATURES])
    scaler = StandardScaler().fit(X)
    pca = PCA(n_components=3, random_state=42).fit(scaler.transform(X))
    return scaler, pca, joblib.load(km_file("kmeans_pca.pkl"))


@st.cache_resource
def train(df, clf):
    """clf=True: phân loại mua lại; False: hồi quy chi tiêu. Trả về (scaler, mô hình tuyến tính, cây, y_test, dự đoán)."""
    t = LABEL if clf else TARGET
    X_tr, X_te, y_tr, y_te = train_test_split(df[FEATURES], df[t], test_size=0.2, random_state=42,
                                              stratify=df[t] if clf else None)
    sc = StandardScaler().fit(X_tr)
    if clf:
        lin = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
        names = ["Hồi quy Logistic", "Cây quyết định"]
    else:
        lin, tree = LinearRegression(), DecisionTreeRegressor(max_depth=5, random_state=42)
        names = ["Hồi quy tuyến tính", "Cây quyết định"]
    lin.fit(sc.transform(X_tr), y_tr)
    tree.fit(X_tr, y_tr)
    return sc, lin, tree, y_te, dict(zip(names, [lin.predict(sc.transform(X_te)), tree.predict(X_te)]))


df = load_data()

# ---------- Nhập số liệu một khách hàng ----------
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
    m4.metric("Tỉ lệ mua lại 90 ngày", f"{df[LABEL].mean():.1%}")

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

# ---------- Tab 1: KMeans ----------
with tab1:
    thieu = [f for f in KM_FILES if km_file(f) is None]
    if thieu:
        st.error("Thiếu file KMeans: " + ", ".join(thieu) +
                 ". Hãy đặt chúng vào thư mục ket_qua/ nằm cạnh file này (hoặc ngay cạnh file này).")
    else:
        st.subheader("So sánh KMeans trên 7 đặc trưng (log + scale) và trên PCA")
        st.dataframe(pd.read_csv(km_file("comparison.csv")).round(4))

        c1, c2 = st.columns(2)
        c1.image(str(km_file("kmeans_elbow.png")))
        c2.image(str(km_file("kmeans_silhouette.png")))

        c3, c4 = st.columns(2)
        c3.write("Điểm theo K - PCA")
        c3.dataframe(pd.read_csv(km_file("scores_pca.csv")).round(4))
        c4.write("Điểm theo K - log + scale")
        c4.dataframe(pd.read_csv(km_file("scores_logscaled.csv")).round(4))

        st.subheader("Đặc điểm trung bình từng cụm (K = 2)")
        profile = pd.read_csv(km_file("cluster_profile.csv"), index_col=0, encoding="utf-8-sig")
        st.dataframe(profile)
        st.image(str(km_file("kmeans_clusters.png")))

        st.subheader("Khách hàng vừa nhập thuộc cụm nào?")
        if st.button("Phân cụm", key="btn_cluster"):
            scaler_k, pca_k, model = load_kmeans(df)
            z = pca_k.transform(scaler_k.transform(log_features(new_row[KM_FEATURES])))
            label = int(model.predict(z)[0])
            st.success(f"Khách hàng này thuộc **Cụm {label}**")
            st.write("Hồ sơ trung bình của cụm này:")
            st.dataframe(profile.loc[[label]])


# ---------- Tab 2 & 3: dùng chung một hàm ----------
def model_tab(clf):
    sc, lin, tree, y, pred = train(df, clf)
    st.subheader("So sánh 2 mô hình (tập kiểm tra 20%)")
    if clf:
        fns = [accuracy_score, precision_score, recall_score, f1_score]
        st.dataframe(pd.DataFrame({n: [f(y, p) for f in fns] for n, p in pred.items()},
                                  index=["Độ chính xác (Accuracy)", "Precision (mua lại)",
                                         "Recall (mua lại)", "F1 (mua lại)"]).round(3))
        st.subheader("Ma trận nhầm lẫn (hàng = thực tế, cột = dự đoán)")
        for col, (n, p) in zip(st.columns(2), pred.items()):
            col.write(n)
            col.dataframe(pd.DataFrame(confusion_matrix(y, p), index=["Thực: Không mua lại", "Thực: Mua lại"],
                                       columns=["Đoán: Không mua lại", "Đoán: Mua lại"]))
    else:
        st.dataframe(pd.DataFrame([{"Mô hình": n, "RMSE (GBP)": np.sqrt(mean_squared_error(y, p)),
                                    "R2": r2_score(y, p)} for n, p in pred.items()]).round(4))
        n = st.selectbox("Xem biểu đồ thực tế vs dự đoán của mô hình", list(pred))
        st.scatter_chart(pd.DataFrame({"Thực tế": y.to_numpy(), "Dự đoán": pred[n]}), x="Thực tế", y="Dự đoán")

    st.subheader("Khách vừa nhập có mua lại trong 90 ngày không?" if clf
                 else "Dự đoán chi tiêu 90 ngày tới của khách hàng vừa nhập")
    if st.button("Dự đoán mua lại" if clf else "Dự đoán chi tiêu", key=f"btn_{clf}"):
        f = (lambda m, x: m.predict_proba(x)[0, 1]) if clf else (lambda m, x: max(m.predict(x)[0], 0))
        for col, n, v in zip(st.columns(2), pred, [f(lin, sc.transform(new_row)), f(tree, new_row)]):
            if clf:
                col.metric(n, "Mua lại" if v >= 0.5 else "Không mua lại", f"xác suất {v:.0%}", delta_color="off")
            else:
                col.metric(n, f"{v:,.2f} GBP")
        if not clf:
            st.caption("Giá trị âm được đưa về 0.")


with tab2:
    model_tab(False)
with tab3:
    model_tab(True)
