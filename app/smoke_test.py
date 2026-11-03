"""Integration smoke test: launch the Streamlit script and persist one inference in Qdrant."""
from pathlib import Path
from streamlit.testing.v1 import AppTest
from database import check_connection, history

assert check_connection(), "Qdrant database is unreachable"
for path in ("data_khach_hang.csv", "kmeans_pca.pkl", "comparison.csv",
             "scores_pca.csv", "scores_logscaled.csv", "cluster_profile.csv",
             "kmeans_elbow.png", "kmeans_silhouette.png", "kmeans_clusters.png"):
    assert (Path(__file__).parent / path).is_file(), f"Missing application asset: {path}"

app = AppTest.from_file(str(Path(__file__).parent / "AI.py"), default_timeout=120)
app.run()
assert not app.exception, f"Initial Streamlit render failed: {app.exception}"
app.button(key="all_three").click().run()
assert not app.exception, f"Three-model inference failed: {app.exception}"
records = history()
assert not records.empty, "No inference saved to Qdrant"
for column in ("cluster", "predicted_spend_gbp", "repurchase_probability", "repurchase_label"):
    assert column in records.columns, f"Missing saved output: {column}"
print("PASS: Streamlit rendered, all 3 predictions ran, and Qdrant persisted an inference")
