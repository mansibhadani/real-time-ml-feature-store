"""
Offline Spark ETL → Online Feature Store Pipeline
===================================================
Simulates the offline→online feature pipeline pattern used in production ML systems.
Reads raw event data, computes aggregated features, pushes to Redis online store.

Architecture:
  S3 (raw events) → Spark (batch aggregation) → Redis (online serving)
                                              ↘ Parquet (offline store)

Author: Mansi Bhadani
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_STORE_URL = "http://localhost:8000"


def create_spark_session():
    return (
        SparkSession.builder
        .appName("FeatureStore-ETL")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )


def generate_sample_events(spark: SparkSession):
    """Simulates reading from S3 data lake / Kafka topic."""
    schema = StructType([
        StructField("user_id", StringType()),
        StructField("content_id", StringType()),
        StructField("event_type", StringType()),   # watch | search | click
        StructField("genre", StringType()),
        StructField("duration_min", DoubleType()),
        StructField("device_type", StringType()),
        StructField("rating", DoubleType()),
        StructField("event_ts", TimestampType()),
    ])

    from datetime import datetime, timedelta
    import random

    users    = [f"user_{i:04d}" for i in range(1, 51)]
    contents = [f"content_{i:04d}" for i in range(1, 101)]
    genres   = ["action", "drama", "comedy"]
    devices  = ["mobile", "desktop", "tv"]
    now      = datetime.now()

    rows = []
    for _ in range(5000):
        rows.append((
            random.choice(users),
            random.choice(contents),
            random.choice(["watch", "search", "click"]),
            random.choice(genres),
            round(random.uniform(1, 120), 1),
            random.choice(devices),
            round(random.uniform(1, 5), 1),
            now - timedelta(days=random.randint(0, 7), hours=random.randint(0, 23))
        ))

    return spark.createDataFrame(rows, schema)


def compute_user_features(events_df):
    """
    Aggregate raw events into user-level features.
    Window: last 7 days. Mirrors production feature computation.
    """
    watch_events = events_df.filter(F.col("event_type") == "watch")

    watch_agg = watch_events.groupBy("user_id").agg(
        F.sum("duration_min").alias("watch_hours_7d_raw"),
        F.count("*").alias("watch_count_7d"),
        F.avg("duration_min").alias("avg_session_duration_min"),
        F.mode("device_type").alias("device_type"),
    )

    # Genre affinities — fraction of watch time per genre
    genre_agg = watch_events.groupBy("user_id", "genre").agg(
        F.sum("duration_min").alias("genre_watch_min")
    )
    total_watch = genre_agg.groupBy("user_id").agg(
        F.sum("genre_watch_min").alias("total_watch_min")
    )
    genre_agg = genre_agg.join(total_watch, "user_id")
    genre_agg = genre_agg.withColumn(
        "genre_fraction", F.col("genre_watch_min") / F.col("total_watch_min")
    )

    action_aff = genre_agg.filter(F.col("genre") == "action").select(
        "user_id", F.col("genre_fraction").alias("genre_affinity_action"))
    drama_aff = genre_agg.filter(F.col("genre") == "drama").select(
        "user_id", F.col("genre_fraction").alias("genre_affinity_drama"))
    comedy_aff = genre_agg.filter(F.col("genre") == "comedy").select(
        "user_id", F.col("genre_fraction").alias("genre_affinity_comedy"))

    search_freq = events_df.filter(F.col("event_type") == "search")\
        .groupBy("user_id").agg(F.count("*").alias("search_frequency_7d"))

    user_features = (
        watch_agg
        .join(action_aff, "user_id", "left").fillna(0, ["genre_affinity_action"])
        .join(drama_aff,  "user_id", "left").fillna(0, ["genre_affinity_drama"])
        .join(comedy_aff, "user_id", "left").fillna(0, ["genre_affinity_comedy"])
        .join(search_freq,"user_id", "left").fillna(0, ["search_frequency_7d"])
        .withColumn("watch_hours_7d", F.col("watch_hours_7d_raw") / 60.0)
        .drop("watch_hours_7d_raw")
    )

    logger.info(f"Computed features for {user_features.count()} users")
    return user_features


def compute_content_features(events_df):
    """Aggregate content-level features for the online feature store."""
    content_features = events_df.groupBy("content_id", "genre").agg(
        F.count(F.when(F.col("event_type") == "watch",  1)).alias("watch_count"),
        F.count(F.when(F.col("event_type") == "click",  1)).alias("click_count"),
        F.avg(F.when(F.col("rating").isNotNull(), F.col("rating"))).alias("avg_rating"),
        F.max("event_ts").alias("last_seen_ts"),
    )

    max_watches = content_features.agg(F.max("watch_count")).collect()[0][0]
    from datetime import datetime
    now_ts = datetime.now().timestamp()

    content_features = (
        content_features
        .withColumn("popularity_score", F.col("watch_count") / max_watches)
        .withColumn("age_days",
            (F.lit(now_ts) - F.col("last_seen_ts").cast("long")) / 86400.0)
        .withColumn("recency_score", F.greatest(F.lit(0.0), F.lit(1.0) - F.col("age_days") / 30.0))
        .withColumn("genre_action", (F.col("genre") == "action").cast("double"))
        .withColumn("genre_drama",  (F.col("genre") == "drama").cast("double"))
        .withColumn("genre_comedy", (F.col("genre") == "comedy").cast("double"))
        .fillna(3.0, ["avg_rating"])
    )

    logger.info(f"Computed features for {content_features.count()} content items")
    return content_features


def push_to_feature_store(df, entity_type: str):
    """
    Push Spark DataFrame rows to Redis online feature store via REST SDK.
    In production: use direct Redis bulk-write (pipeline) for throughput.
    """
    endpoint = f"{FEATURE_STORE_URL}/features/{entity_type}"
    rows = df.collect()
    success, failed = 0, 0

    for row in rows:
        try:
            payload = row.asDict()
            # Clean up non-serializable types
            for k, v in payload.items():
                if hasattr(v, 'isoformat'):
                    payload[k] = v.isoformat()
                elif v is None:
                    payload[k] = 0.0

            resp = requests.post(endpoint, json=payload, timeout=2)
            if resp.status_code == 200:
                success += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            logger.warning(f"Failed to push {entity_type} feature: {e}")

    logger.info(f"Feature store push [{entity_type}]: {success} success, {failed} failed")


def save_offline_store(df, path: str):
    """Save to Parquet as offline/historical feature store (for training)."""
    (df.write
       .mode("overwrite")
       .partitionBy("user_id" if "user_id" in df.columns else "content_id")
       .parquet(path))
    logger.info(f"Saved offline features to {path}")


def run_pipeline():
    """Main ETL pipeline entry point."""
    spark = create_spark_session()
    logger.info("Pipeline started — reading raw events...")

    events = generate_sample_events(spark)

    logger.info("Computing user features...")
    user_features = compute_user_features(events)

    logger.info("Computing content features...")
    content_features = compute_content_features(events)

    logger.info("Saving offline feature stores (Parquet)...")
    save_offline_store(user_features,    "/tmp/offline_store/user_features")
    save_offline_store(content_features, "/tmp/offline_store/content_features")

    logger.info("Pushing to online feature store (Redis)...")
    push_to_feature_store(user_features,    "user")
    push_to_feature_store(content_features, "content")

    logger.info("Pipeline complete!")
    spark.stop()


if __name__ == "__main__":
    run_pipeline()
