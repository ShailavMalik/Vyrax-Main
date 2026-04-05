"""Real-time emotion detection with trigger-driven heavy inference."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import numpy as np

from config import (
    BENCHMARK_REPORT_PATH,
    CAMERA_INDEX,
    DEFAULT_BENCHMARK_SECONDS,
    DETECTION_FRAME_SIZE,
    DISPLAY_FRAME_WIDTH,
    ENABLE_JSONL_LOGGING,
    EMOTION_COLORS,
    FEATURE_BASELINE_ALPHA,
    FONT,
    FONT_SCALE,
    HEAVY_DETECTION_INTERVAL_SECONDS,
    JSONL_LOG_PATH,
    LANDMARK_CHANGE_THRESHOLD,
    SHOW_FPS,
    TEXT_COLOR,
    TEXT_THICKNESS,
    BG_COLOR,
    BG_PADDING,
    BOX_HOLD_MISSING_SECONDS,
    BOX_SMOOTHING_ALPHA,
    BOX_THICKNESS,
    DEFAULT_EMOTION_COLOR,
    FACE_MOVEMENT_THRESHOLD_PX,
)
from emotion_engine import (
    DetectionCache,
    EmotionIntelligenceEngine,
    cache_last_result,
    compute_face_quality,
    detect_emotion_fer,
    detect_faces_mediapipe,
    extract_face,
    extract_features_mediapipe,
    face_movement_detected,
    get_face_mesh_config,
    landmark_changed,
    landmark_signature,
    preprocess_face,
    should_run_detection,
)
from utils import draw_text_with_bg, resize_to_width

LOGGER = logging.getLogger("realtime_emotion")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

STATUS_TRACKING = "Tracking"
STATUS_HEAVY = "Heavy inference"
STATUS_NO_FACE = "No face"
STATUS_UNCERTAIN = "uncertain"


def scale_box_to_original(
    small_box: Tuple[int, int, int, int],
    small_shape: Tuple[int, int],
    orig_shape: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    """Scale a bounding box from detection frame to full-resolution frame."""
    sx1, sy1, sx2, sy2 = small_box
    small_h, small_w = small_shape
    orig_h, orig_w = orig_shape

    if small_w <= 0 or small_h <= 0:
        return None

    scale_x = orig_w / float(small_w)
    scale_y = orig_h / float(small_h)

    x1 = int(sx1 * scale_x)
    y1 = int(sy1 * scale_y)
    x2 = int(sx2 * scale_x)
    y2 = int(sy2 * scale_y)

    x1 = max(0, min(x1, orig_w - 1))
    y1 = max(0, min(y1, orig_h - 1))
    x2 = max(0, min(x2, orig_w - 1))
    y2 = max(0, min(y2, orig_h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def draw_results(frame: np.ndarray, result: Dict[str, Any], fps: float) -> None:
    """Render tracked face and key status lines."""
    box = result.get("box")
    final_emotion = str(result.get("final_emotion", STATUS_UNCERTAIN))
    final_confidence = float(result.get("final_confidence", 0.0))
    raw_emotion = str(result.get("raw_emotion", STATUS_UNCERTAIN))
    raw_confidence = float(result.get("raw_confidence", 0.0))
    smoothed_emotion = str(result.get("smoothed_emotion", STATUS_UNCERTAIN))
    smoothed_confidence = float(result.get("smoothed_confidence", 0.0))
    status = str(result.get("status", STATUS_TRACKING))

    color = EMOTION_COLORS.get(final_emotion, DEFAULT_EMOTION_COLOR)

    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        label = f"{final_emotion.upper()} {final_confidence * 100:.0f}%"
        draw_text_with_bg(
            frame,
            label,
            x1 + 4,
            max(22, y1 - 8),
            FONT,
            FONT_SCALE,
            TEXT_COLOR,
            TEXT_THICKNESS,
            color,
            BG_PADDING,
        )

    panel_lines = [
        f"Status: {status}",
        f"Raw: {raw_emotion.upper()} {raw_confidence * 100:.0f}%",
        f"Smoothed: {smoothed_emotion.upper()} {smoothed_confidence * 100:.0f}%",
        f"Final: {final_emotion.upper()} {final_confidence * 100:.0f}%",
    ]

    triggers = result.get("rule_triggers") or []
    if triggers:
        panel_lines.append(f"Rules: {', '.join(triggers)}")

    y = 30
    for line in panel_lines:
        draw_text_with_bg(
            frame,
            line,
            10,
            y,
            FONT,
            FONT_SCALE,
            TEXT_COLOR,
            TEXT_THICKNESS,
            BG_COLOR,
            BG_PADDING,
        )
        y += 26

    if SHOW_FPS:
        draw_text_with_bg(
            frame,
            f"FPS: {fps:.1f}",
            10,
            y,
            FONT,
            FONT_SCALE,
            (0, 255, 0),
            TEXT_THICKNESS,
            BG_COLOR,
            BG_PADDING,
        )


@dataclass
class PipelineState:
    cache: DetectionCache
    frame_index: int = 0
    fps: float = 0.0
    frame_count: int = 0
    last_fps_time: float = 0.0
    display_box: Optional[Tuple[int, int, int, int]] = None
    last_display_box_ts: float = 0.0


@dataclass
class BenchmarkStats:
    total_frames: int = 0
    heavy_frames: int = 0
    tracking_frames: int = 0
    no_face_frames: int = 0
    movement_triggers: int = 0
    landmark_triggers: int = 0
    time_triggers: int = 0
    emotion_switches: int = 0
    uncertain_frames: int = 0
    avg_final_conf_sum: float = 0.0
    avg_raw_conf_sum: float = 0.0
    _last_final: Optional[str] = None

    def record(self, result: Dict[str, Any]) -> None:
        self.total_frames += 1
        status = str(result.get("status", ""))
        if status == STATUS_HEAVY:
            self.heavy_frames += 1
        elif status == STATUS_TRACKING:
            self.tracking_frames += 1
        elif status == STATUS_NO_FACE:
            self.no_face_frames += 1

        if bool(result.get("movement_trigger")):
            self.movement_triggers += 1
        if bool(result.get("landmark_trigger")):
            self.landmark_triggers += 1
        if bool(result.get("time_trigger")):
            self.time_triggers += 1

        final_emotion = str(result.get("final_emotion", STATUS_UNCERTAIN))
        if final_emotion == STATUS_UNCERTAIN:
            self.uncertain_frames += 1

        if self._last_final is not None and final_emotion != self._last_final and final_emotion != STATUS_UNCERTAIN:
            self.emotion_switches += 1
        self._last_final = final_emotion

        self.avg_final_conf_sum += float(result.get("final_confidence", 0.0))
        self.avg_raw_conf_sum += float(result.get("raw_confidence", 0.0))

    def to_dict(self, elapsed_seconds: float) -> Dict[str, Any]:
        frames = max(self.total_frames, 1)
        return {
            "elapsed_seconds": elapsed_seconds,
            "total_frames": self.total_frames,
            "fps_effective": self.total_frames / max(elapsed_seconds, 1e-6),
            "heavy_frames": self.heavy_frames,
            "tracking_frames": self.tracking_frames,
            "no_face_frames": self.no_face_frames,
            "heavy_rate": self.heavy_frames / frames,
            "movement_triggers": self.movement_triggers,
            "landmark_triggers": self.landmark_triggers,
            "time_triggers": self.time_triggers,
            "emotion_switches": self.emotion_switches,
            "uncertain_rate": self.uncertain_frames / frames,
            "avg_final_confidence": self.avg_final_conf_sum / frames,
            "avg_raw_confidence": self.avg_raw_conf_sum / frames,
        }


class JsonlEventLogger:
    """Writes one JSON event per heavy inference for explainability/audit."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")


class EmotionDetectionPipeline:
    """Main loop: lightweight tracking every frame, heavy FER on triggers only."""

    def __init__(self, benchmark_seconds: int = 0, benchmark_report: Optional[str] = None) -> None:
        self.cap: Optional[cv2.VideoCapture] = None
        self.state = PipelineState(cache=DetectionCache(), last_fps_time=time.time())
        self.face_mesh_config = get_face_mesh_config()
        self.engine = EmotionIntelligenceEngine()
        self.feature_baseline: Optional[Dict[str, float]] = None
        self.benchmark_seconds = max(0, int(benchmark_seconds))
        self.benchmark_report = benchmark_report or BENCHMARK_REPORT_PATH
        self.benchmark_stats = BenchmarkStats()
        self.session_start = time.time()
        self.event_logger = JsonlEventLogger(JSONL_LOG_PATH) if ENABLE_JSONL_LOGGING else None

    def initialize_camera(self) -> bool:
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            print("ERROR: Could not open webcam")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        print("Camera initialized")
        return True

    def update_fps(self) -> float:
        now = time.time()
        elapsed = now - self.state.last_fps_time

        self.state.frame_count += 1
        if elapsed >= 1.0:
            self.state.fps = self.state.frame_count / elapsed
            self.state.frame_count = 0
            self.state.last_fps_time = now

        return self.state.fps

    def _no_face_result(self) -> Dict[str, Any]:
        # Keep the last box briefly to avoid rapid rectangle blinking on short misses.
        hold_box: Optional[Tuple[int, int, int, int]] = None
        now = time.time()
        if self.state.display_box is not None and (now - self.state.last_display_box_ts) <= BOX_HOLD_MISSING_SECONDS:
            hold_box = self.state.display_box

        return {
            "status": STATUS_NO_FACE,
            "box": hold_box,
            "raw_emotion": STATUS_UNCERTAIN,
            "raw_confidence": 0.0,
            "smoothed_emotion": STATUS_UNCERTAIN,
            "smoothed_confidence": 0.0,
            "final_emotion": STATUS_UNCERTAIN,
            "final_confidence": 0.0,
            "rule_triggers": [],
            "scores": {},
        }

    def _smooth_display_box(self, box: Tuple[int, int, int, int], now_ts: float) -> Tuple[int, int, int, int]:
        """Apply EMA smoothing to reduce box jitter in the UI."""
        prev = self.state.display_box
        if prev is None:
            self.state.display_box = box
            self.state.last_display_box_ts = now_ts
            return box

        alpha = max(0.0, min(1.0, float(BOX_SMOOTHING_ALPHA)))
        smoothed = (
            int((alpha * prev[0]) + ((1.0 - alpha) * box[0])),
            int((alpha * prev[1]) + ((1.0 - alpha) * box[1])),
            int((alpha * prev[2]) + ((1.0 - alpha) * box[2])),
            int((alpha * prev[3]) + ((1.0 - alpha) * box[3])),
        )

        self.state.display_box = smoothed
        self.state.last_display_box_ts = now_ts
        return smoothed

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        if frame is None or frame.size == 0:
            return self._no_face_result()

        current_time = time.time()
        self.state.frame_index += 1

        orig_h, orig_w = frame.shape[:2]
        small_w, small_h = DETECTION_FRAME_SIZE
        small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)

        detections = detect_faces_mediapipe(small)
        if not detections:
            return self._no_face_result()

        sx1, sy1, sx2, sy2, det_score = detections[0]
        box_orig = scale_box_to_original((sx1, sy1, sx2, sy2), (small_h, small_w), (orig_h, orig_w))
        if box_orig is None:
            return self._no_face_result()

        # Lightweight per-frame tracking signal.
        tracking_crop = extract_face(small, (sx1, sy1, sx2, sy2), face_size=(128, 128), padding=8)
        tracking_features = extract_features_mediapipe(tracking_crop, self.face_mesh_config) if tracking_crop is not None else {}
        self._update_feature_baseline(tracking_features)
        current_signature = landmark_signature(tracking_features)

        movement = face_movement_detected(self.state.cache.last_box, box_orig, FACE_MOVEMENT_THRESHOLD_PX)
        lm_change = landmark_changed(
            self.state.cache.last_landmark_signature,
            current_signature,
            threshold=LANDMARK_CHANGE_THRESHOLD,
        )

        run_heavy = should_run_detection(
            current_time,
            self.state.cache,
            self.state.frame_index,
            time_interval_seconds=HEAVY_DETECTION_INTERVAL_SECONDS,
            face_moved=movement,
            landmarks_changed=lm_change,
        )
        time_trigger = (current_time - self.state.cache.last_detection_time) >= HEAVY_DETECTION_INTERVAL_SECONDS

        if not run_heavy and self.state.cache.result is not None:
            cached = dict(self.state.cache.result)
            cached["status"] = STATUS_TRACKING
            cached["box"] = self._smooth_display_box(box_orig, current_time)
            cached["movement_trigger"] = False
            cached["landmark_trigger"] = False
            cached["time_trigger"] = False
            return cached

        face_crop = extract_face(frame, box_orig)
        if face_crop is None:
            return self._no_face_result()

        # Heavy pass uses normalized crop for FER and geometric features.
        normalized_crop = preprocess_face(face_crop)
        if normalized_crop is None:
            return self._no_face_result()

        fer_emotion, fer_scores = detect_emotion_fer(normalized_crop)
        if fer_emotion is None or not fer_scores:
            uncertain = {
                "status": STATUS_HEAVY,
                "box": box_orig,
                "detector_confidence": float(det_score),
                "raw_emotion": STATUS_UNCERTAIN,
                "raw_confidence": 0.0,
                "smoothed_emotion": STATUS_UNCERTAIN,
                "smoothed_confidence": 0.0,
                "final_emotion": STATUS_UNCERTAIN,
                "final_confidence": 0.0,
                "rule_triggers": [],
                "scores": {},
            }
            cache_last_result(self.state.cache, uncertain, current_time, self.state.frame_index, box_orig, current_signature)
            return uncertain

        features = extract_features_mediapipe(normalized_crop, self.face_mesh_config)
        face_quality = compute_face_quality(normalized_crop)
        intelligence = self.engine.evaluate(fer_scores, features, current_time, face_quality=face_quality)

        result = {
            "status": STATUS_HEAVY,
            "box": self._smooth_display_box(box_orig, current_time),
            "detector_confidence": float(det_score),
            "raw_emotion": intelligence["raw_emotion"],
            "raw_confidence": float(intelligence["raw_confidence"]),
            "smoothed_emotion": intelligence["smoothed_emotion"] or STATUS_UNCERTAIN,
            "smoothed_confidence": float(intelligence["smoothed_confidence"]),
            "final_emotion": intelligence["final_emotion"],
            "final_confidence": float(intelligence["final_confidence"]),
            "rule_triggers": intelligence.get("rule_triggers", []),
            "scores": intelligence.get("scores", {}),
            "features": features,
            "movement_trigger": movement,
            "landmark_trigger": lm_change,
            "time_trigger": bool(time_trigger and not movement and not lm_change),
            "face_quality": face_quality,
            "feature_baseline": self.feature_baseline,
            "hold_remaining": intelligence.get("hold_remaining", 0.0),
        }

        cache_last_result(
            self.state.cache,
            result,
            current_time,
            self.state.frame_index,
            current_box=box_orig,
            current_landmark_signature=current_signature,
        )

        if self.event_logger is not None:
            self.event_logger.log(
                {
                    "ts": current_time,
                    "status": STATUS_HEAVY,
                    "raw_scores": result["scores"],
                    "rule_triggers": result["rule_triggers"],
                    "final_emotion": result["final_emotion"],
                    "final_confidence": result["final_confidence"],
                    "fps": self.state.fps,
                    "movement_trigger": result["movement_trigger"],
                    "landmark_trigger": result["landmark_trigger"],
                    "time_trigger": result["time_trigger"],
                    "face_quality": result["face_quality"],
                }
            )

        LOGGER.info(
            "FER raw=%s triggers=%s final=%s(%.2f) fps=%.1f",
            result["scores"],
            result["rule_triggers"],
            result["final_emotion"],
            result["final_confidence"],
            self.state.fps,
        )
        return result

    def render(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        draw_results(frame, result, self.update_fps())
        return resize_to_width(frame, DISPLAY_FRAME_WIDTH)

    def run(self) -> None:
        if self.cap is None:
            print("ERROR: Camera is not initialized")
            return

        print("Press 'q' to quit")

        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("ERROR: Failed to read frame")
                break

            result = self.process_frame(frame)
            self.benchmark_stats.record(result)
            annotated = self.render(frame, result)
            cv2.imshow("Real-Time Emotion Detection", annotated)

            if self.benchmark_seconds > 0 and (time.time() - self.session_start) >= self.benchmark_seconds:
                print("Benchmark duration reached. Stopping run.")
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self._save_benchmark_report()
        self.cleanup()

    def _update_feature_baseline(self, features: Dict[str, Any]) -> None:
        if not features.get("available"):
            return

        sample = {
            "mouth_open_ratio": float(features.get("mouth_open_ratio", 0.0)),
            "eye_open_ratio": float(features.get("eye_open_ratio", 0.0)),
            "eyebrow_position": float(features.get("eyebrow_position", 0.0)),
            "eyebrow_inner_raise": float(features.get("eyebrow_inner_raise", 0.0)),
            "smile_ratio": float(features.get("smile_ratio", 0.0)),
        }

        if self.feature_baseline is None:
            self.feature_baseline = dict(sample)
            return

        alpha = max(0.0, min(1.0, FEATURE_BASELINE_ALPHA))
        for key, value in sample.items():
            self.feature_baseline[key] = (alpha * self.feature_baseline[key]) + ((1.0 - alpha) * value)

    def _save_benchmark_report(self) -> None:
        elapsed = max(0.0, time.time() - self.session_start)
        summary = self.benchmark_stats.to_dict(elapsed)

        if self.benchmark_seconds > 0:
            report_path = Path(self.benchmark_report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"Benchmark report saved to: {report_path}")

        print("Benchmark summary:")
        print(json.dumps(summary, indent=2))

    def cleanup(self) -> None:
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time emotion detection")
    parser.add_argument(
        "--benchmark-seconds",
        type=int,
        default=0,
        help=f"Run for fixed seconds and save report (default off; suggested {DEFAULT_BENCHMARK_SECONDS})",
    )
    parser.add_argument(
        "--benchmark-report",
        type=str,
        default=BENCHMARK_REPORT_PATH,
        help="Path to benchmark JSON report",
    )
    args = parser.parse_args()

    pipeline = EmotionDetectionPipeline(
        benchmark_seconds=args.benchmark_seconds,
        benchmark_report=args.benchmark_report,
    )
    if not pipeline.initialize_camera():
        return

    try:
        pipeline.run()
    except KeyboardInterrupt:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
