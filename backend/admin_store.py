from __future__ import annotations

import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str) and value.strip():
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(raw).astimezone(timezone.utc)
        except ValueError:
            return utc_now()
    return utc_now()


class AdminStore:
    def __init__(self) -> None:
        self.mongo_uri = os.getenv("MONGODB_URI", "").strip()
        self.mongo_db = os.getenv("MONGODB_DB", "vyrax_analytics").strip() or "vyrax_analytics"
        self.snapshot_high_confidence = float(os.getenv("SNAPSHOT_HIGH_CONFIDENCE", "0.85"))
        self.snapshot_min_interval_seconds = max(1, int(os.getenv("SNAPSHOT_MIN_INTERVAL_SECONDS", "15")))
        self.snapshot_high_conf_interval_seconds = max(
            self.snapshot_min_interval_seconds,
            int(os.getenv("SNAPSHOT_HIGH_CONFIDENCE_INTERVAL_SECONDS", "45")),
        )
        self.snapshot_keepalive_interval_seconds = max(
            self.snapshot_high_conf_interval_seconds,
            int(os.getenv("SNAPSHOT_KEEPALIVE_INTERVAL_SECONDS", "120")),
        )
        self._client: Optional[MongoClient] = None
        self._db = None

    @property
    def configured(self) -> bool:
        return bool(self.mongo_uri)

    def _ensure_connected(self) -> None:
        if not self.configured:
            raise RuntimeError("Missing MONGODB_URI")
        if self._client is None:
            self._client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=3000)
            self._client.admin.command("ping")
            self._db = self._client[self.mongo_db]
            self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.snapshots.create_index([("sessionId", ASCENDING), ("timestamp", DESCENDING)])
        self.emotions.create_index([("sessionId", ASCENDING), ("timestamp", DESCENDING)])
        self.sessions.create_index([("sessionId", ASCENDING)], unique=True)

    @property
    def snapshots(self) -> Collection:
        self._ensure_connected()
        return self._db["snapshots"]

    @property
    def emotions(self) -> Collection:
        self._ensure_connected()
        return self._db["emotions"]

    @property
    def sessions(self) -> Collection:
        self._ensure_connected()
        return self._db["sessions"]

    def health(self) -> Dict[str, Any]:
        if not self.configured:
            return {"configured": False, "connected": False}
        try:
            self._ensure_connected()
            return {"configured": True, "connected": True, "database": self.mongo_db}
        except Exception as exc:
            return {"configured": True, "connected": False, "error": str(exc)}

    def should_store_snapshot(
        self,
        session_id: str,
        emotion: str,
        confidence: float,
        timestamp: Optional[Any] = None,
    ) -> Dict[str, Any]:
        high_conf = float(confidence) >= self.snapshot_high_confidence
        now_ts = parse_timestamp(timestamp)
        latest = self.snapshots.find_one(
            {"sessionId": session_id},
            projection={"emotion": 1, "timestamp": 1, "_id": 0},
            sort=[("timestamp", DESCENDING)],
        )
        if latest is None:
            return {"store": True, "reason": "first_snapshot"}

        latest_ts = parse_timestamp(latest.get("timestamp"))
        elapsed_seconds = max(0.0, (now_ts - latest_ts).total_seconds())
        changed = str(latest.get("emotion", "")).lower() != str(emotion).lower()

        if changed:
            if elapsed_seconds < self.snapshot_min_interval_seconds:
                return {"store": False, "reason": "cooldown_emotion_change"}
            return {"store": True, "reason": "emotion_change"}

        if high_conf:
            if elapsed_seconds >= self.snapshot_high_conf_interval_seconds:
                return {"store": True, "reason": "periodic_high_confidence"}
            return {"store": False, "reason": "cooldown_high_confidence"}

        if elapsed_seconds >= self.snapshot_keepalive_interval_seconds:
            return {"store": True, "reason": "periodic_keepalive"}

        return {"store": False, "reason": "gated"}

    def insert_emotion(self, *, session_id: str, emotion: str, confidence: float, timestamp: Any) -> Dict[str, Any]:
        ts = parse_timestamp(timestamp)
        doc = {
            "sessionId": session_id,
            "emotion": emotion,
            "confidence": float(confidence),
            "timestamp": ts,
        }
        self.emotions.insert_one(doc)

        self._upsert_session_summary(session_id=session_id)
        doc.pop("_id", None)
        return doc

    def insert_snapshot(
        self,
        *,
        session_id: str,
        emotion: str,
        confidence: float,
        timestamp: Any,
        image_url: str,
        blob_name: str,
        content_type: str,
        size_bytes: int,
    ) -> Dict[str, Any]:
        ts = parse_timestamp(timestamp)
        doc = {
            "sessionId": session_id,
            "imageUrl": image_url,
            "emotion": emotion,
            "confidence": float(confidence),
            "timestamp": ts,
            "blobName": blob_name,
            "contentType": content_type,
            "sizeBytes": int(size_bytes),
        }
        self.snapshots.insert_one(doc)
        doc.pop("_id", None)
        return doc

    def _upsert_session_summary(self, *, session_id: str) -> Dict[str, Any]:
        cursor = self.emotions.find(
            {"sessionId": session_id},
            projection={"emotion": 1, "confidence": 1, "timestamp": 1, "_id": 0},
            sort=[("timestamp", ASCENDING)],
        )
        events = list(cursor)
        total_events = len(events)
        if total_events == 0:
            return {}

        start_time = events[0]["timestamp"]
        end_time = events[-1]["timestamp"]

        counts = Counter(str(evt.get("emotion", "uncertain")) for evt in events)
        dominant_emotion = counts.most_common(1)[0][0] if counts else "uncertain"

        confidence_sum = sum(float(evt.get("confidence", 0.0) or 0.0) for evt in events)
        avg_confidence = confidence_sum / total_events

        summary = {
            "sessionId": session_id,
            "startTime": start_time,
            "endTime": end_time,
            "totalEvents": total_events,
            "dominantEmotion": dominant_emotion,
            "avgConfidence": avg_confidence,
            "updatedAt": utc_now(),
        }

        self.sessions.update_one({"sessionId": session_id}, {"$set": summary}, upsert=True)
        return summary

    def list_snapshots(self, *, session_id: Optional[str], limit: int = 100) -> List[Dict[str, Any]]:
        query = {"sessionId": session_id} if session_id else {}
        docs = list(
            self.snapshots.find(query, projection={"_id": 0}).sort("timestamp", DESCENDING).limit(limit)
        )
        return docs

    def list_emotions(self, *, session_id: Optional[str], limit: int = 500) -> List[Dict[str, Any]]:
        query = {"sessionId": session_id} if session_id else {}
        docs = list(
            self.emotions.find(query, projection={"_id": 0}).sort("timestamp", DESCENDING).limit(limit)
        )
        return docs

    def get_summary(self, *, session_id: Optional[str]) -> Any:
        if session_id:
            summary = self._upsert_session_summary(session_id=session_id)
            return summary

        docs = list(self.sessions.find({}, projection={"_id": 0}).sort("endTime", DESCENDING).limit(50))
        return docs
