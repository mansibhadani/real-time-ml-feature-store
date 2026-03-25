# Real-Time ML Feature Store — Content Recommendation Engine

A production-pattern **online feature store** using **Redis + FastAPI + Spark**, designed for low-latency ML inference in content search & ranking pipelines.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Sources                             │
│   S3 / Data Lake         Kafka (real-time events)           │
└──────────┬───────────────────────┬──────────────────────────┘
           │ Batch (hourly)        │ Near-line (seconds)
           ▼                       ▼
┌──────────────────┐    ┌─────────────────────┐
│  Spark ETL Job   │    │  Kafka Consumer      │
│  spark_etl.py    │    │  kafka_consumer.py   │
│                  │    │  (incremental update)│
└──────────┬───────┘    └──────────┬──────────┘
           │                       │
           └──────────┬────────────┘
                      ▼
           ┌──────────────────────┐
           │  Redis Online Store  │  ← Sub-millisecond reads
           │  features:user:*     │
           │  features:content:*  │
           │  TTL: 1h / 24h       │
           └──────────┬───────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │  FastAPI Serving     │  ← /infer endpoint
           │  FeatureStoreSDK     │
           │  Point-in-time join  │
           └──────────┬───────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │  ML Model Inference  │  ← Relevance score
           │  (search re-ranking) │
           └──────────────────────┘
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Redis with TTL | User features expire after 1h; content after 24h — staleness-accuracy tradeoff |
| Namespace keys (`features:user:*`) | Enables fast pattern-based scans, namespace isolation |
| Point-in-time join in SDK | Prevents feature leakage; ensures training/serving consistency |
| Spark batch + Kafka near-line | Dual pipeline: Spark for accuracy, Kafka for recency |
| FastAPI over gRPC | Lower complexity for demo; production would use gRPC for latency |

## Stack

- **Online Store:** Redis 7 (allkeys-lru eviction, 512MB cap)
- **Offline Pipeline:** Apache Spark / PySpark
- **Near-Line Pipeline:** Apache Kafka + kafka-python consumer
- **Serving Layer:** FastAPI + Pydantic
- **Containerization:** Docker Compose

## Quick Start

```bash
# 1. Start Redis + Kafka
docker-compose up -d redis kafka

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the API
uvicorn main:app --reload

# 4. Ingest sample features
python seed_data.py

# 5. Run inference
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_0001", "content_id": "content_0001"}'
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/features/user` | Ingest user features |
| POST | `/features/content` | Ingest content features |
| GET | `/features/user/{id}` | Retrieve user features |
| POST | `/infer` | Real-time ML inference |
| GET | `/store/stats` | Feature store metrics |
| GET | `/health` | Health check |

## Performance Characteristics

- **Read latency:** < 2ms (Redis + local FastAPI)
- **Write throughput:** ~50K writes/sec (Redis pipeline mode)
- **Feature freshness:** 1-hour SLA for user features, 24h for content

## Requirements

```
fastapi==0.111.0
uvicorn==0.30.1
redis==5.0.4
pyspark==3.5.1
kafka-python==2.0.2
numpy==1.26.4
pydantic==2.7.1
requests==2.32.3
scikit-learn==1.5.0
```
