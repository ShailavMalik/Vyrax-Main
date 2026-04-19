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
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from pathlib import Path
import sys

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
from flask import Flask, Response, jsonify, stream_with_context
from flask_cors import CORS

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from .realtime_emotion import EmotionDetectionPipeline, STATUS_NO_FACE, STATUS_UNCERTAIN
except ImportError:  # pragma: no cover - script execution path
    from realtime_emotion import EmotionDetectionPipeline, STATUS_NO_FACE, STATUS_UNCERTAIN


HOST = "0.0.0.0"
PORT = 8000


@dataclass
class SharedState:
    latest_result: Dict[str, Any] = field(default_factory=dict)
    latest_frame: Optional[bytes] = None
    camera_ready: bool = False
    last_error: str = ""
    last_updated: float = 0.0


pipeline: Optional[EmotionDetectionPipeline] = None
state = SharedState()
state_lock = threading.Lock()
stop_event = threading.Event()


def _make_placeholder_frame(message: str) -> bytes:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (1279, 719), (32, 32, 32), thickness=-1)
    cv2.putText(frame, "Vyra-X Camera", (80, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, message, (80, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Check camera permissions or the camera index in backend/config.py", (80, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
    ok, buffer = cv2.imencode(".jpg", frame)
    return buffer.tobytes() if ok else b""


def _encode_frame(frame: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame)
    return buffer.tobytes() if ok else b""


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

        result = pipeline.process_frame(frame)
        annotated = pipeline.render(frame, result)
        jpeg = _encode_frame(annotated)

        with state_lock:
            state.latest_result = dict(result)
            state.latest_result["camera_ready"] = True
            state.latest_result["error"] = ""
            state.latest_frame = jpeg if jpeg else _make_placeholder_frame("Unable to encode frame")
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
    return jsonify(payload)


@app.get("/video_feed")
def video_feed():
    return Response(
        stream_with_context(_mjpeg_stream()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


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