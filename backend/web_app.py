"""Flask web server for the Vyra-X React UI.

This server keeps the camera feed and the emotion JSON endpoint alive for the
frontend. If the webcam cannot be opened, it serves a visible placeholder frame
instead of failing with a black screen or exiting the process.
"""

from __future__ import annotations

import os
import threading
import time
import logging
import warnings
import base64
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from pathlib import Path
import sys
from datetime import datetime, timezone

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency at import time
    load_dotenv = None

# Keep runtime logs concise for launcher mode while preserving errors.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings(
    "ignore",
    message=r".*tf\.losses\.sparse_softmax_cross_entropy is deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*You are sending unauthenticated requests to the HF Hub.*",
    category=UserWarning,
)

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("detection").setLevel(logging.WARNING)

try:
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.set_stderrthreshold("error")
except Exception:
    pass

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

REPO_ROOT = BACKEND_DIR.parent
if load_dotenv is not None:
    load_dotenv(REPO_ROOT / ".env", override=False)
    load_dotenv(BACKEND_DIR / ".env", override=False)

try:
    from .realtime_emotion import EmotionDetectionPipeline, STATUS_NO_FACE, STATUS_UNCERTAIN
except ImportError:  # pragma: no cover - script execution path
    from realtime_emotion import EmotionDetectionPipeline, STATUS_NO_FACE, STATUS_UNCERTAIN

try:
    from .snapshot_storage import AsyncCleanupScheduler, SnapshotRepository, build_container_client
except ImportError:  # pragma: no cover - script execution path
    from snapshot_storage import AsyncCleanupScheduler, SnapshotRepository, build_container_client

try:
    from .admin_store import AdminStore, parse_timestamp
except ImportError:  # pragma: no cover - script execution path
    from admin_store import AdminStore, parse_timestamp


HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING", "")
AZURE_BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER", "vyra-snapshots")
SNAPSHOT_BLOB_PREFIX = os.getenv("SNAPSHOT_BLOB_PREFIX", "snapshots")
SNAPSHOT_CLEANUP_EVERY_UPLOADS = max(1, int(os.getenv("SNAPSHOT_CLEANUP_EVERY_UPLOADS", "1")))
MJPEG_QUALITY = max(45, min(95, int(os.getenv("MJPEG_QUALITY", "72"))))


@dataclass
class SharedState:
    latest_result: Dict[str, Any] = field(default_factory=dict)
    latest_frame: Optional[bytes] = None
    latest_raw_frame: Optional[np.ndarray] = None
    camera_ready: bool = False
    last_error: str = ""
    last_updated: float = 0.0


pipeline: Optional[EmotionDetectionPipeline] = None
state = SharedState()
state_lock = threading.Lock()
stop_event = threading.Event()
snapshot_container_client = None
snapshot_repo: Optional[SnapshotRepository] = None
snapshot_cleanup_scheduler = AsyncCleanupScheduler(run_every_uploads=SNAPSHOT_CLEANUP_EVERY_UPLOADS)
admin_store = AdminStore()
SESSION_ID = os.getenv("SESSION_ID", f"vyrax-session-{uuid.uuid4().hex[:8]}")
snapshot_persist_lock = threading.Lock()
ui_camera_enabled = True
MIN_NON_BLACK_LUMA = float(os.getenv("SNAPSHOT_MIN_LUMA", "8.0"))
MIN_FACE_QUALITY_FOR_SNAPSHOT = float(os.getenv("SNAPSHOT_MIN_FACE_QUALITY", "0.08"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _frame_is_mostly_black(frame_bgr: Optional[np.ndarray]) -> bool:
    if frame_bgr is None or frame_bgr.size == 0:
        return True
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray)) < MIN_NON_BLACK_LUMA
    except Exception:
        return True


def _decode_image_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    if not image_bytes:
        return None
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _result_has_face(result: Dict[str, Any]) -> bool:
    status = str(result.get("status") or "").strip().lower()
    if status == STATUS_NO_FACE.lower():
        return False

    box = result.get("box")
    has_box = isinstance(box, (list, tuple)) and len(box) == 4 and any(float(v) > 0.0 for v in box)

    try:
        face_quality = float(result.get("face_quality") or 0.0)
    except Exception:
        face_quality = 0.0

    return has_box or face_quality >= MIN_FACE_QUALITY_FOR_SNAPSHOT


def _state_has_recent_face(max_age_seconds: float = 2.0) -> bool:
    now = time.time()
    with state_lock:
        latest = dict(state.latest_result or {})
        updated = float(state.last_updated or 0.0)

    if not latest or (now - updated) > max_age_seconds:
        return False
    return _result_has_face(latest)


def _make_placeholder_frame(message: str) -> bytes:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (1279, 719), (32, 32, 32), thickness=-1)
    cv2.putText(frame, "Vyra-X Camera", (80, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, message, (80, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Check camera permissions or the camera index in backend/config.py", (80, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
    ok, buffer = cv2.imencode(".jpg", frame)
    return buffer.tobytes() if ok else b""


def _encode_frame(frame: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), MJPEG_QUALITY])
    return buffer.tobytes() if ok else b""


def _parse_data_url(payload: str) -> tuple[str, bytes]:
    content_type = "image/jpeg"
    b64_payload = payload
    if payload.startswith("data:") and "," in payload:
        header, b64_payload = payload.split(",", 1)
        if ";" in header:
            content_type = header[5:].split(";", 1)[0] or content_type

    image_bytes = base64.b64decode(b64_payload, validate=False)
    if not image_bytes:
        raise ValueError("Empty image payload")
    return content_type, image_bytes


def _extension_for_content_type(content_type: str) -> str:
    mapping = {
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
    }
    return mapping.get(content_type.lower(), "jpg")


def _get_snapshot_services():
    global snapshot_container_client, snapshot_repo

    if not AZURE_BLOB_CONNECTION_STRING:
        raise RuntimeError("Missing AZURE_BLOB_CONNECTION_STRING")

    if snapshot_container_client is None:
        snapshot_container_client = build_container_client(
            connection_string=AZURE_BLOB_CONNECTION_STRING,
            container_name=AZURE_BLOB_CONTAINER,
        )

    if snapshot_repo is None:
        snapshot_repo = SnapshotRepository(BACKEND_DIR / "logs" / "snapshots.db")

    return snapshot_container_client, snapshot_repo


def _persist_emotion_and_snapshot(
    *,
    frame_bgr: np.ndarray,
    session_id: str,
    emotion: str,
    confidence: float,
    timestamp: datetime,
    gate: Dict[str, Any],
) -> None:
    try:
        with snapshot_persist_lock:
            admin_store.insert_emotion(
                session_id=session_id,
                emotion=emotion,
                confidence=confidence,
                timestamp=timestamp,
            )

            if not gate.get("store"):
                return

            content_type = "image/jpeg"
            image_bytes = _encode_frame(frame_bgr)
            if not image_bytes:
                raise RuntimeError("Failed to encode snapshot frame")

            container_client, repo = _get_snapshot_services()
            snapshot_cleanup_scheduler.schedule(
                container_client=container_client,
                snapshot_repo=repo,
                logger=logging.getLogger("web_app"),
            )

            ext = _extension_for_content_type(content_type)
            prefix = SNAPSHOT_BLOB_PREFIX.strip("/")
            blob_name = f"{prefix}/{timestamp:%Y/%m/%d}/{uuid.uuid4().hex}.{ext}"

            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(
                image_bytes,
                overwrite=False,
                content_type=content_type,
            )

            blob_url = blob_client.url
            repo.upsert_snapshot(
                blob_name=blob_name,
                blob_url=blob_url,
                content_length=len(image_bytes),
            )

            admin_store.insert_snapshot(
                session_id=session_id,
                emotion=emotion,
                confidence=confidence,
                timestamp=timestamp,
                image_url=blob_url,
                blob_name=blob_name,
                content_type=content_type,
                size_bytes=len(image_bytes),
            )
    except Exception:
        logging.getLogger("web_app").exception("Failed to persist emotion/snapshot event")


def _capture_loop() -> None:
    global pipeline
    read_failures = 0

    if pipeline is None:
        pipeline = EmotionDetectionPipeline()

    while not stop_event.is_set():
        if pipeline.cap is None:
            if not pipeline.initialize_camera():
                with state_lock:
                    state.camera_ready = False
                    state.last_error = "Could not open a usable webcam"
                    state.latest_frame = _make_placeholder_frame(state.last_error)
                    state.latest_result = {
                        "status": STATUS_NO_FACE,
                        "box": None,
                        "raw_emotion": STATUS_UNCERTAIN,
                        "raw_confidence": 0.0,
                        "smoothed_emotion": STATUS_UNCERTAIN,
                        "smoothed_confidence": 0.0,
                        "decision_source": "CAMERA_UNAVAILABLE",
                        "geometry_emotion": "none",
                        "geometry_strength": 0.0,
                        "geometry_reason": "camera_unavailable",
                        "expression_intensity": 0.0,
                        "final_emotion": STATUS_UNCERTAIN,
                        "final_confidence": 0.0,
                        "rule_triggers": [],
                        "scores": {},
                        "movement_trigger": False,
                        "landmark_trigger": False,
                        "time_trigger": False,
                        "face_quality": 0.0,
                        "camera_ready": False,
                        "error": state.last_error,
                    }
                    state.last_updated = time.time()
                time.sleep(1.5)
                continue

            with state_lock:
                state.camera_ready = True
                state.last_error = ""
            read_failures = 0

        if pipeline.cap is not None and not pipeline.cap.isOpened():
            pipeline.cap.release()
            pipeline.cap = None
            continue

        ok, frame = pipeline.cap.read()
        if not ok or frame is None or frame.size == 0:
            read_failures += 1
            with state_lock:
                state.camera_ready = False
                state.last_error = "Failed to read a webcam frame"
                state.latest_frame = _make_placeholder_frame(state.last_error)
                state.latest_result = {
                    "status": STATUS_NO_FACE,
                    "box": None,
                    "raw_emotion": STATUS_UNCERTAIN,
                    "raw_confidence": 0.0,
                    "smoothed_emotion": STATUS_UNCERTAIN,
                    "smoothed_confidence": 0.0,
                    "decision_source": "CAMERA_READ_FAILED",
                    "geometry_emotion": "none",
                    "geometry_strength": 0.0,
                    "geometry_reason": "camera_read_failed",
                    "expression_intensity": 0.0,
                    "final_emotion": STATUS_UNCERTAIN,
                    "final_confidence": 0.0,
                    "rule_triggers": [],
                    "scores": {},
                    "movement_trigger": False,
                    "landmark_trigger": False,
                    "time_trigger": False,
                    "face_quality": 0.0,
                    "camera_ready": False,
                    "error": state.last_error,
                }
                state.last_updated = time.time()

            # After a sustained read failure streak, force camera reopen.
            if read_failures >= 25:
                if pipeline.cap is not None:
                    pipeline.cap.release()
                pipeline.cap = None
                read_failures = 0

            time.sleep(0.5)
            continue

        read_failures = 0

        loop_started = time.time()
        result = pipeline.process_frame(frame)
        annotated = pipeline.render(frame, result)
        jpeg = _encode_frame(annotated)
        result["fps"] = float(getattr(pipeline.state, "fps", 0.0))
        result["processing_ms"] = float((time.time() - loop_started) * 1000.0)

        snapshot_gate: Dict[str, Any] = {"store": False, "reason": "disabled"}
        if not ui_camera_enabled:
            snapshot_gate = {"store": False, "reason": "camera_disabled_ui"}
        elif result.get("status") == "Heavy inference":
            if not _result_has_face(result):
                snapshot_gate = {"store": False, "reason": "no_face_detected"}
            elif _frame_is_mostly_black(frame):
                snapshot_gate = {"store": False, "reason": "dark_frame"}
            else:
                try:
                    snapshot_gate = admin_store.should_store_snapshot(
                        session_id=SESSION_ID,
                        emotion=str(result.get("final_emotion") or STATUS_UNCERTAIN),
                        confidence=float(result.get("final_confidence") or 0.0),
                        timestamp=datetime.now(timezone.utc),
                    )
                except Exception as exc:
                    snapshot_gate = {"store": False, "reason": f"mongo_gate_failed:{exc}"}

            snapshot_payload = {
                "session_id": SESSION_ID,
                "emotion": str(result.get("final_emotion") or STATUS_UNCERTAIN),
                "confidence": float(result.get("final_confidence") or 0.0),
                "timestamp": datetime.now(timezone.utc),
                "gate": dict(snapshot_gate),
                "frame_bgr": frame.copy(),
            }
            threading.Thread(
                target=_persist_emotion_and_snapshot,
                kwargs=snapshot_payload,
                name="snapshot-persist",
                daemon=True,
            ).start()

        result["snapshot_taken"] = bool(snapshot_gate.get("store"))
        result["snapshot_reason"] = str(snapshot_gate.get("reason", "gated"))
        result["snapshot_session_id"] = SESSION_ID
        result["snapshot_taken_at"] = time.time() if result["snapshot_taken"] else None
        result["snapshot_pending"] = result["snapshot_taken"]

        with state_lock:
            state.latest_result = dict(result)
            state.latest_result["camera_ready"] = True
            state.latest_result["error"] = ""
            state.latest_result["ui_camera_enabled"] = bool(ui_camera_enabled)
            state.latest_frame = jpeg if jpeg else _make_placeholder_frame("Unable to encode frame")
            state.latest_raw_frame = frame.copy()
            state.camera_ready = True
            state.last_error = ""
            state.last_updated = time.time()


def _mjpeg_stream():
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    last_sent = 0.0

    while not stop_event.is_set():
        with state_lock:
            frame = state.latest_frame
            updated = state.last_updated
            camera_ready = state.camera_ready
            error = state.last_error

        if frame is None:
            message = error or ("Camera connected" if camera_ready else "Waiting for webcam...")
            frame = _make_placeholder_frame(message)
            # Ensure the stream starts immediately even before camera loop updates.
            updated = time.time()

        if updated == last_sent:
            time.sleep(0.04)
            continue

        last_sent = updated
        yield boundary + frame + b"\r\n"


app = Flask(__name__)
CORS(app)

# Reduce noisy per-request logs (e.g. repeated /emotion polling) in launcher output.
logging.getLogger("werkzeug").setLevel(logging.WARNING)
logging.getLogger("flask.app").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
app.logger.setLevel(logging.WARNING)


@app.get("/health")
def health():
    with state_lock:
        return jsonify(
            {
                "ok": True,
                "camera_ready": state.camera_ready,
                "last_error": state.last_error,
                "updated": state.last_updated,
                "mongo": admin_store.health(),
            }
        )


@app.get("/emotion")
def emotion():
    with state_lock:
        payload = dict(state.latest_result)
        if not payload:
            payload = {
                "status": STATUS_NO_FACE,
                "box": None,
                "raw_emotion": STATUS_UNCERTAIN,
                "raw_confidence": 0.0,
                "smoothed_emotion": STATUS_UNCERTAIN,
                "smoothed_confidence": 0.0,
                "decision_source": "WAITING_FOR_CAMERA",
                "geometry_emotion": "none",
                "geometry_strength": 0.0,
                "geometry_reason": "waiting_for_camera",
                "expression_intensity": 0.0,
                "final_emotion": STATUS_UNCERTAIN,
                "final_confidence": 0.0,
                "rule_triggers": [],
                "scores": {},
            }

        payload.setdefault("camera_ready", state.camera_ready)
        payload.setdefault("error", state.last_error)
        payload.setdefault("updated", state.last_updated)
        payload.setdefault("ui_camera_enabled", bool(ui_camera_enabled))
    return jsonify(payload)


@app.post("/api/camera-state")
def api_camera_state():
    global ui_camera_enabled

    payload = request.get_json(silent=True) or {}
    enabled = payload.get("enabled")
    if enabled is None:
        return jsonify({"error": "Missing 'enabled' boolean"}), 400

    ui_camera_enabled = bool(enabled)
    return jsonify({"ok": True, "cameraEnabled": ui_camera_enabled})


@app.get("/video_feed")
def video_feed():
    return Response(
        stream_with_context(_mjpeg_stream()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/snapshot")
def upload_snapshot():
    payload = request.get_json(silent=True) or {}
    image_payload = payload.get("image")
    if not image_payload or not isinstance(image_payload, str):
        return jsonify({"error": "Missing 'image' base64 payload"}), 400

    try:
        content_type, image_bytes = _parse_data_url(image_payload)
    except Exception as exc:
        return jsonify({"error": f"Invalid image payload: {exc}"}), 400

    try:
        container_client, repo = _get_snapshot_services()

        snapshot_cleanup_scheduler.schedule(
            container_client=container_client,
            snapshot_repo=repo,
            logger=logging.getLogger("web_app"),
        )

        now = datetime.now(timezone.utc)
        ext = _extension_for_content_type(content_type)
        prefix = SNAPSHOT_BLOB_PREFIX.strip("/")
        blob_name = f"{prefix}/{now:%Y/%m/%d}/{uuid.uuid4().hex}.{ext}"

        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(
            image_bytes,
            overwrite=False,
            content_type=content_type,
        )

        blob_url = blob_client.url
        repo.upsert_snapshot(
            blob_name=blob_name,
            blob_url=blob_url,
            content_length=len(image_bytes),
        )

        return jsonify(
            {
                "ok": True,
                "blob_name": blob_name,
                "url": blob_url,
                "content_type": content_type,
                "size_bytes": len(image_bytes),
            }
        ), 201
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"Snapshot upload failed: {exc}"}), 500


@app.post("/api/upload-snapshot")
def api_upload_snapshot():
    payload = request.get_json(silent=True) or {}
    image_payload = payload.get("image")
    if not image_payload or not isinstance(image_payload, str):
        return jsonify({"error": "Missing 'image' base64 payload"}), 400

    session_id = str(payload.get("sessionId") or payload.get("session_id") or "default")
    emotion = str(payload.get("emotion") or STATUS_UNCERTAIN)
    confidence = float(payload.get("confidence") or 0.0)
    timestamp = parse_timestamp(payload.get("timestamp"))

    if not ui_camera_enabled:
        return jsonify(
            {
                "ok": True,
                "stored": False,
                "reason": "camera_disabled_ui",
                "sessionId": session_id,
            }
        ), 202

    try:
        content_type, image_bytes = _parse_data_url(image_payload)
    except Exception as exc:
        return jsonify({"error": f"Invalid image payload: {exc}"}), 400

    frame_bgr = _decode_image_bytes(image_bytes)
    if _frame_is_mostly_black(frame_bgr):
        return jsonify(
            {
                "ok": True,
                "stored": False,
                "reason": "dark_frame",
                "sessionId": session_id,
            }
        ), 202

    if not _state_has_recent_face():
        return jsonify(
            {
                "ok": True,
                "stored": False,
                "reason": "no_face_detected",
                "sessionId": session_id,
            }
        ), 202

    try:
        gate = admin_store.should_store_snapshot(
            session_id=session_id,
            emotion=emotion,
            confidence=confidence,
            timestamp=timestamp,
        )
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"MongoDB unavailable: {exc}"}), 503

    if not gate.get("store"):
        return jsonify(
            {
                "ok": True,
                "stored": False,
                "reason": gate.get("reason", "gated"),
                "sessionId": session_id,
            }
        ), 202

    try:
        container_client, repo = _get_snapshot_services()
        snapshot_cleanup_scheduler.schedule(
            container_client=container_client,
            snapshot_repo=repo,
            logger=logging.getLogger("web_app"),
        )

        now = datetime.now(timezone.utc)
        ext = _extension_for_content_type(content_type)
        prefix = SNAPSHOT_BLOB_PREFIX.strip("/")
        blob_name = f"{prefix}/{now:%Y/%m/%d}/{uuid.uuid4().hex}.{ext}"

        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(
            image_bytes,
            overwrite=False,
            content_type=content_type,
        )

        blob_url = blob_client.url
        repo.upsert_snapshot(
            blob_name=blob_name,
            blob_url=blob_url,
            content_length=len(image_bytes),
        )

        admin_store.insert_snapshot(
            session_id=session_id,
            emotion=emotion,
            confidence=confidence,
            timestamp=timestamp,
            image_url=blob_url,
            blob_name=blob_name,
            content_type=content_type,
            size_bytes=len(image_bytes),
        )

        return jsonify(
            {
                "ok": True,
                "stored": True,
                "reason": gate.get("reason", "first_snapshot"),
                "sessionId": session_id,
                "url": blob_url,
                "blob_name": blob_name,
                "emotion": emotion,
                "confidence": confidence,
                "timestamp": timestamp.isoformat(),
            }
        ), 201
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"Snapshot upload failed: {exc}"}), 500


@app.post("/api/manual-snapshot")
def api_manual_snapshot():
    payload = request.get_json(silent=True) or {}
    session_id = str(payload.get("sessionId") or payload.get("session_id") or SESSION_ID)
    emotion = str(payload.get("emotion") or STATUS_UNCERTAIN)
    confidence = float(payload.get("confidence") or 0.0)
    timestamp = parse_timestamp(payload.get("timestamp"))

    if not ui_camera_enabled:
        return jsonify(
            {
                "ok": True,
                "stored": False,
                "reason": "camera_disabled_ui",
                "sessionId": session_id,
            }
        ), 202

    with state_lock:
        frame_bgr = None if state.latest_raw_frame is None else state.latest_raw_frame.copy()
        latest_result = dict(state.latest_result or {})

    if frame_bgr is None:
        return jsonify({"error": "No live frame available"}), 503

    if _frame_is_mostly_black(frame_bgr):
        return jsonify(
            {
                "ok": True,
                "stored": False,
                "reason": "dark_frame",
                "sessionId": session_id,
            }
        ), 202

    if not _result_has_face(latest_result):
        return jsonify(
            {
                "ok": True,
                "stored": False,
                "reason": "no_face_detected",
                "sessionId": session_id,
            }
        ), 202

    try:
        gate = admin_store.should_store_snapshot(
            session_id=session_id,
            emotion=emotion,
            confidence=confidence,
            timestamp=timestamp,
        )
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"MongoDB unavailable: {exc}"}), 503

    if not gate.get("store"):
        return jsonify(
            {
                "ok": True,
                "stored": False,
                "reason": gate.get("reason", "gated"),
                "sessionId": session_id,
            }
        ), 202

    try:
        content_type = "image/jpeg"
        image_bytes = _encode_frame(frame_bgr)
        if not image_bytes:
            raise RuntimeError("Failed to encode snapshot frame")

        container_client, repo = _get_snapshot_services()
        snapshot_cleanup_scheduler.schedule(
            container_client=container_client,
            snapshot_repo=repo,
            logger=logging.getLogger("web_app"),
        )

        ext = _extension_for_content_type(content_type)
        prefix = SNAPSHOT_BLOB_PREFIX.strip("/")
        blob_name = f"{prefix}/{timestamp:%Y/%m/%d}/{uuid.uuid4().hex}.{ext}"

        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(
            image_bytes,
            overwrite=False,
            content_type=content_type,
        )

        blob_url = blob_client.url
        repo.upsert_snapshot(
            blob_name=blob_name,
            blob_url=blob_url,
            content_length=len(image_bytes),
        )

        admin_store.insert_snapshot(
            session_id=session_id,
            emotion=emotion,
            confidence=confidence,
            timestamp=timestamp,
            image_url=blob_url,
            blob_name=blob_name,
            content_type=content_type,
            size_bytes=len(image_bytes),
        )

        return jsonify(
            {
                "ok": True,
                "stored": True,
                "reason": gate.get("reason", "manual"),
                "sessionId": session_id,
                "url": blob_url,
                "blob_name": blob_name,
                "emotion": emotion,
                "confidence": confidence,
                "timestamp": timestamp.isoformat(),
                "manual": True,
            }
        ), 201
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"Manual snapshot failed: {exc}"}), 500


@app.post("/api/emotion")
def api_store_emotion():
    payload = request.get_json(silent=True) or {}
    session_id = str(payload.get("sessionId") or payload.get("session_id") or "default")
    emotion = str(payload.get("emotion") or STATUS_UNCERTAIN)
    confidence = float(payload.get("confidence") or 0.0)
    timestamp = parse_timestamp(payload.get("timestamp"))

    try:
        inserted = admin_store.insert_emotion(
            session_id=session_id,
            emotion=emotion,
            confidence=confidence,
            timestamp=timestamp,
        )
        return jsonify({"ok": True, "event": _json_safe(inserted)}), 201
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"Emotion store failed: {exc}"}), 500


@app.get("/api/snapshots")
def api_get_snapshots():
    session_id = request.args.get("sessionId")
    limit = int(request.args.get("limit") or "100")
    limit = max(1, min(limit, 1000))

    try:
        data = admin_store.list_snapshots(session_id=session_id, limit=limit)
        return jsonify({"ok": True, "count": len(data), "items": _json_safe(data)})
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"Snapshot fetch failed: {exc}"}), 500


@app.get("/api/emotions")
def api_get_emotions():
    session_id = request.args.get("sessionId")
    limit = int(request.args.get("limit") or "500")
    limit = max(1, min(limit, 5000))

    try:
        data = admin_store.list_emotions(session_id=session_id, limit=limit)
        return jsonify({"ok": True, "count": len(data), "items": _json_safe(data)})
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"Emotion fetch failed: {exc}"}), 500


@app.get("/api/summary")
def api_get_summary():
    session_id = request.args.get("sessionId")

    try:
        data = admin_store.get_summary(session_id=session_id)
        return jsonify({"ok": True, "summary": _json_safe(data)})
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"Summary fetch failed: {exc}"}), 500


def main() -> None:
    worker = threading.Thread(target=_capture_loop, name="camera-capture", daemon=True)
    worker.start()
    try:
        app.run(host=HOST, port=PORT, debug=False, threaded=True, use_reloader=False)
    finally:
        stop_event.set()
        if pipeline.cap is not None:
            pipeline.cleanup()


if __name__ == "__main__":
    main()