"""
Near-Line Feature Updater — Kafka Consumer
===========================================
Consumes real-time user interaction events from Kafka,
computes incremental feature updates, and writes to Redis feature store.

Architecture mirrors Disney+'s near-line pipeline pattern:
  Kafka (user events) → Consumer → Incremental feature update → Redis

Author: Mansi Bhadani
"""

from kafka import KafkaConsumer
import redis
import json
import logging
import time
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP    = "localhost:9092"
KAFKA_TOPIC        = "user-interactions"
KAFKA_GROUP_ID     = "feature-store-updater"
REDIS_HOST         = "localhost"
REDIS_PORT         = 6379
FEATURE_TTL        = 3600   # 1 hour


class NearLineFeatureUpdater:
    """
    Consumes Kafka events and applies incremental feature updates to Redis.
    Uses Redis HINCRBY / pipeline for atomic, low-latency writes.
    """

    def __init__(self):
        self.r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
        self.consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id=KAFKA_GROUP_ID,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )
        self.buffer     = defaultdict(list)
        self.buffer_size = 100   # Micro-batch size
        self.flush_interval_sec = 5

    def _key(self, user_id: str) -> str:
        return f"features:user:{user_id}"

    def process_event(self, event: dict):
        """
        Incrementally update features based on event type.
        event schema: {user_id, event_type, genre, duration_min, device_type, ts}
        """
        user_id    = event.get("user_id")
        event_type = event.get("event_type")
        genre      = event.get("genre", "")
        duration   = float(event.get("duration_min", 0))
        device     = event.get("device_type", "mobile")

        if not user_id:
            return

        key  = self._key(user_id)
        pipe = self.r.pipeline()

        if event_type == "watch":
            # Increment watch stats atomically
            pipe.hincrbyfloat(key, "watch_hours_7d", duration / 60.0)
            pipe.hincrbyfloat(key, "avg_session_duration_min", duration)
            if genre in ("action", "drama", "comedy"):
                pipe.hincrbyfloat(key, f"genre_affinity_{genre}", 0.05)

        elif event_type == "search":
            pipe.hincrby(key, "search_frequency_7d", 1)

        # Refresh TTL on every interaction
        pipe.expire(key, FEATURE_TTL)
        pipe.hset(key, "device_type", device)
        pipe.hset(key, "last_updated", time.time())
        pipe.execute()

    def run(self):
        logger.info(f"Starting near-line feature updater | topic={KAFKA_TOPIC}")
        last_flush = time.time()

        try:
            for message in self.consumer:
                event = message.value
                self.process_event(event)

                # Log throughput every 1000 messages
                if message.offset % 1000 == 0:
                    logger.info(f"Processed offset={message.offset} | partition={message.partition}")

        except KeyboardInterrupt:
            logger.info("Shutting down near-line updater...")
        finally:
            self.consumer.close()


# ── Kafka Producer (for local testing) ───────────────────────────────────────
def run_test_producer():
    """Generates synthetic Kafka messages for local testing."""
    from kafka import KafkaProducer
    import random

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    users   = [f"user_{i:04d}" for i in range(1, 20)]
    genres  = ["action", "drama", "comedy"]
    devices = ["mobile", "desktop", "tv"]
    types   = ["watch", "search", "click"]

    for i in range(500):
        event = {
            "user_id":      random.choice(users),
            "event_type":   random.choice(types),
            "genre":        random.choice(genres),
            "duration_min": round(random.uniform(5, 90), 1),
            "device_type":  random.choice(devices),
            "ts":           time.time()
        }
        producer.send(KAFKA_TOPIC, event)
        if i % 100 == 0:
            logger.info(f"Produced {i} events...")
        time.sleep(0.01)

    producer.flush()
    logger.info("Test producer done.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "producer":
        run_test_producer()
    else:
        updater = NearLineFeatureUpdater()
        updater.run()
