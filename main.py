"""
Real-Time ML Feature Store Demo
================================
Stack: Redis (online store) + FastAPI (serving layer) + Scikit-learn (model)
Use Case: Content recommendation score prediction (Disney+ style)
Author: Mansi Bhadani
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import json
import pickle
import numpy as np
from typing import Optional
import time

app = FastAPI(
    title="ML Feature Store API",
    description="Low-latency online feature serving for real-time ML inference",
    version="1.0.0"
)

# ── Redis Connection ──────────────────────────────────────────────────────────
r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# ── Schemas ───────────────────────────────────────────────────────────────────
class UserFeatures(BaseModel):
    user_id: str
    watch_hours_7d: float
    genre_affinity_action: float
    genre_affinity_drama: float
    genre_affinity_comedy: float
    search_frequency_7d: int
    avg_session_duration_min: float
    device_type: str  # mobile | desktop | tv

class ContentFeatures(BaseModel):
    content_id: str
    popularity_score: float
    recency_score: float       # 1.0 = new, 0.0 = old
    genre_action: float
    genre_drama: float
    genre_comedy: float
    avg_rating: float

class InferenceRequest(BaseModel):
    user_id: str
    content_id: str

class InferenceResponse(BaseModel):
    user_id: str
    content_id: str
    relevance_score: float
    features_used: dict
    latency_ms: float
    cache_hit: bool


# ── Feature Store SDK ─────────────────────────────────────────────────────────
class FeatureStoreSDK:
    """
    SDK wrapping Redis for low-latency online feature access.
    Mirrors production patterns: TTL-based expiry, JSON serialization,
    namespace isolation per entity type.
    """
    USER_TTL = 3600       # 1 hour — user features refresh hourly
    CONTENT_TTL = 86400   # 24 hours — content features refresh daily

    def __init__(self, redis_client: redis.Redis):
        self.r = redis_client

    def _user_key(self, user_id: str) -> str:
        return f"features:user:{user_id}"

    def _content_key(self, content_id: str) -> str:
        return f"features:content:{content_id}"

    def write_user_features(self, features: UserFeatures) -> bool:
        """Write user features with TTL. Called by offline ETL / Spark job."""
        key = self._user_key(features.user_id)
        payload = features.model_dump()
        payload["written_at"] = time.time()
        self.r.setex(key, self.USER_TTL, json.dumps(payload))
        return True

    def write_content_features(self, features: ContentFeatures) -> bool:
        """Write content features with TTL. Called by batch pipeline."""
        key = self._content_key(features.content_id)
        payload = features.model_dump()
        payload["written_at"] = time.time()
        self.r.setex(key, self.CONTENT_TTL, json.dumps(payload))
        return True

    def get_user_features(self, user_id: str) -> Optional[dict]:
        """Sub-millisecond feature retrieval from Redis."""
        raw = self.r.get(self._user_key(user_id))
        return json.loads(raw) if raw else None

    def get_content_features(self, content_id: str) -> Optional[dict]:
        """Sub-millisecond feature retrieval from Redis."""
        raw = self.r.get(self._content_key(content_id))
        return json.loads(raw) if raw else None

    def get_feature_vector(self, user_id: str, content_id: str) -> Optional[np.ndarray]:
        """
        Point-in-time correct feature join for inference.
        Returns merged feature vector ready for model input.
        """
        u = self.get_user_features(user_id)
        c = self.get_content_features(content_id)
        if not u or not c:
            return None, u, c

        # Cross features — genre affinity x content genre
        action_match = u["genre_affinity_action"] * c["genre_action"]
        drama_match  = u["genre_affinity_drama"]  * c["genre_drama"]
        comedy_match = u["genre_affinity_comedy"]  * c["genre_comedy"]
        device_enc   = {"mobile": 0, "desktop": 1, "tv": 2}.get(u["device_type"], 0)

        vector = np.array([
            u["watch_hours_7d"] / 100.0,
            u["search_frequency_7d"] / 50.0,
            u["avg_session_duration_min"] / 120.0,
            c["popularity_score"],
            c["recency_score"],
            c["avg_rating"] / 5.0,
            action_match,
            drama_match,
            comedy_match,
            device_enc / 2.0,
        ], dtype=np.float32)

        features_used = {**{f"user.{k}": v for k, v in u.items()},
                         **{f"content.{k}": v for k, v in c.items()}}
        return vector, features_used


# ── Instantiate SDK ───────────────────────────────────────────────────────────
store = FeatureStoreSDK(r)

# ── Simple trained model (logistic regression as placeholder) ─────────────────
def get_model():
    """
    In production: load from S3/SageMaker Model Registry.
    Here: a deterministic scoring function simulating a trained ranker.
    """
    def score(vector: np.ndarray) -> float:
        weights = np.array([0.15, 0.10, 0.08, 0.20, 0.12, 0.18, 0.06, 0.05, 0.04, 0.02])
        raw = float(np.dot(vector, weights))
        return round(min(max(raw, 0.0), 1.0), 4)
    return score

model = get_model()


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.post("/features/user", summary="Ingest user features into online store")
def write_user(features: UserFeatures):
    """ETL pipeline calls this to push user features into Redis feature store."""
    store.write_user_features(features)
    return {"status": "ok", "user_id": features.user_id, "ttl_seconds": FeatureStoreSDK.USER_TTL}


@app.post("/features/content", summary="Ingest content features into online store")
def write_content(features: ContentFeatures):
    """Batch pipeline calls this to push content features into Redis feature store."""
    store.write_content_features(features)
    return {"status": "ok", "content_id": features.content_id, "ttl_seconds": FeatureStoreSDK.CONTENT_TTL}


@app.get("/features/user/{user_id}", summary="Retrieve user features")
def read_user(user_id: str):
    feats = store.get_user_features(user_id)
    if not feats:
        raise HTTPException(404, f"No features found for user {user_id}")
    return feats


@app.get("/features/content/{content_id}", summary="Retrieve content features")
def read_content(content_id: str):
    feats = store.get_content_features(content_id)
    if not feats:
        raise HTTPException(404, f"No features found for content {content_id}")
    return feats


@app.post("/infer", response_model=InferenceResponse, summary="Real-time ML inference with feature store")
def infer(req: InferenceRequest):
    """
    Core inference endpoint:
    1. Fetches user + content features from Redis (sub-ms latency)
    2. Joins into a point-in-time correct feature vector
    3. Runs model inference
    4. Returns relevance score for search re-ranking
    """
    t0 = time.time()
    vector, features_used = store.get_feature_vector(req.user_id, req.content_id)

    if vector is None:
        raise HTTPException(404, "Features not found. Ensure user and content features are ingested first.")

    score = model(vector)
    latency_ms = round((time.time() - t0) * 1000, 2)

    return InferenceResponse(
        user_id=req.user_id,
        content_id=req.content_id,
        relevance_score=score,
        features_used=features_used,
        latency_ms=latency_ms,
        cache_hit=True
    )


@app.get("/health")
def health():
    try:
        r.ping()
        return {"status": "healthy", "redis": "connected"}
    except:
        return {"status": "degraded", "redis": "disconnected"}


@app.get("/store/stats", summary="Feature store statistics")
def store_stats():
    """Monitor feature store: key counts, memory usage."""
    user_keys    = len(r.keys("features:user:*"))
    content_keys = len(r.keys("features:content:*"))
    mem          = r.info("memory")
    return {
        "user_feature_count": user_keys,
        "content_feature_count": content_keys,
        "redis_used_memory_human": mem.get("used_memory_human"),
        "store_type": "Redis Online Feature Store",
    }
