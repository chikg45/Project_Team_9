"""Qdrant persistence for real customer predictions.

The 7-dimensional standardized feature vector is stored with all three
prediction results as payload. This makes the database a real part of the app
and permits future similarity searches on customer behavior.
"""
import os
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models

COLLECTION = os.getenv("QDRANT_COLLECTION", "customer_inference_history")
URL = os.getenv("QDRANT_URL", "http://localhost:6333")
FEATURE_COUNT = 7


def client():
    return QdrantClient(url=URL, timeout=5)


def ensure_collection(db):
    if not db.collection_exists(COLLECTION):
        db.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=FEATURE_COUNT, distance=models.Distance.COSINE),
        )


def save_prediction(features: dict, vector, cluster: int, spend_gbp: float, repurchase_probability: float):
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if len(vector) != FEATURE_COUNT or not np.isfinite(vector).all():
        raise ValueError("Expected a finite 7-dimensional customer feature vector")
    db = client()
    ensure_collection(db)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "features": {k: float(v) for k, v in features.items()},
        "cluster": int(cluster),
        "predicted_spend_gbp": float(spend_gbp),
        "repurchase_probability": float(repurchase_probability),
        "repurchase_label": int(repurchase_probability >= 0.5),
    }
    db.upsert(
        collection_name=COLLECTION,
        points=[models.PointStruct(id=str(uuid4()), vector=vector.tolist(), payload=payload)],
        wait=True,
    )
    return payload


def history(limit=20):
    db = client()
    if not db.collection_exists(COLLECTION):
        return pd.DataFrame()
    points, _ = db.scroll(collection_name=COLLECTION, limit=limit, with_payload=True, with_vectors=False)
    records = [{"id": str(p.id), **(p.payload or {})} for p in points]
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("created_at", ascending=False)


def check_connection():
    return client().get_collections() is not None
