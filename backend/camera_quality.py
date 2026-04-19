"""Camera quality metrics for real-time emotion detection pipelines."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Tuple
import time

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]


def _clip_0_100(value: float) -> float:
    return float(max(0.0, min(100.0, value)))


def _to_gray(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if image is None or getattr(image, "size", 0) == 0:
        return None
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def sharpness_score(image: Optional[np.ndarray]) -> float:
    """Estimate focus quality from Laplacian variance, normalized to 0-100."""
    gray = _to_gray(image)
    if gray is None:
        return 0.0

    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Typical webcam values vary widely; 800 is a practical saturation point.
    return _clip_0_100((variance / 800.0) * 100.0)


def brightness_score(image: Optional[np.ndarray]) -> float:
    """Mean luminance on grayscale, normalized to 0-100."""
    gray = _to_gray(image)
    if gray is None:
        return 0.0
    return _clip_0_100((float(gray.mean()) / 255.0) * 100.0)


def contrast_score(image: Optional[np.ndarray]) -> float:
    """Grayscale contrast via standard deviation, normalized to 0-100."""
    gray = _to_gray(image)
    if gray is None:
        return 0.0

    std_dev = float(gray.std())
    # Most natural webcam scenes cluster below ~64 std.
    return _clip_0_100((std_dev / 64.0) * 100.0)


def noise_score(image: Optional[np.ndarray]) -> float:
    """Estimate visible noise level (higher means noisier), normalized to 0-100."""
    gray = _to_gray(image)
    if gray is None:
        return 0.0

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = cv2.absdiff(gray, blurred)
    noise_level = float(residual.std())
    return _clip_0_100((noise_level / 32.0) * 100.0)


def face_size_ratio(face_bbox: Optional[BBox], frame_shape: Tuple[int, ...]) -> float:
    """Return face area as percentage of frame area (0-100)."""
    if face_bbox is None or len(frame_shape) < 2:
        return 0.0

    x1, y1, x2, y2 = face_bbox
    frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
    if frame_h <= 0 or frame_w <= 0:
        return 0.0

    face_w = max(0, int(x2) - int(x1))
    face_h = max(0, int(y2) - int(y1))
    if face_w == 0 or face_h == 0:
        return 0.0

    ratio = (float(face_w * face_h) / float(frame_w * frame_h)) * 100.0
    return _clip_0_100(ratio)


@dataclass
class FPSTracker:
    """Track live FPS with a short rolling window."""

    window_size: int = 30
    _timestamps: Deque[float] = field(default_factory=deque)

    def update(self) -> float:
        now = time.perf_counter()
        self._timestamps.append(now)

        while len(self._timestamps) > self.window_size:
            self._timestamps.popleft()

        if len(self._timestamps) < 2:
            return 0.0

        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 1e-9:
            return 0.0

        return (len(self._timestamps) - 1) / elapsed


def fps_tracker(window_size: int = 30) -> FPSTracker:
    """Factory helper for live FPS tracking."""
    return FPSTracker(window_size=max(2, int(window_size)))


@dataclass
class DetectionSuccessTracker:
    """Rolling face-detection success percentage tracker."""

    window_size: int = 120
    _history: Deque[int] = field(default_factory=deque)

    def update(self, face_detected: bool) -> float:
        self._history.append(1 if face_detected else 0)
        while len(self._history) > self.window_size:
            self._history.popleft()

        if not self._history:
            return 0.0

        return (sum(self._history) / float(len(self._history))) * 100.0


def detection_success_rate(tracker: DetectionSuccessTracker, face_detected: bool) -> float:
    """Update and return rolling detection success rate (0-100)."""
    return tracker.update(face_detected)


def camera_quality_score(metrics: Dict[str, float]) -> float:
    """Weighted combined quality score from normalized metrics (0-100)."""
    # Sharpness is intentionally highest weight because blur hurts FER reliability most.
    weights = {
        "sharpness": 0.30,
        "brightness": 0.16,
        "contrast": 0.14,
        "fps": 0.16,
        "face_ratio": 0.12,
        "detection_success": 0.12,
    }

    # Convert noisy signal into a quality-friendly term in case caller wants to use it.
    noise_quality = 100.0 - float(metrics.get("noise", 0.0))
    _ = noise_quality  # Kept for potential diagnostics; not used in the weighted score.

    total = 0.0
    for key, weight in weights.items():
        total += _clip_0_100(float(metrics.get(key, 0.0))) * weight
    return _clip_0_100(total)


@dataclass
class CameraQualityAnalyzer:
    """Compute per-frame quality metrics and optional session aggregates."""

    fps_tracker: FPSTracker = field(default_factory=FPSTracker)
    detection_tracker: DetectionSuccessTracker = field(default_factory=DetectionSuccessTracker)
    compute_every_n_frames: int = 3

    _frame_counter: int = 0
    _last_metrics: Dict[str, float] = field(default_factory=lambda: {
        "sharpness": 0.0,
        "brightness": 0.0,
        "contrast": 0.0,
        "noise": 0.0,
        "fps": 0.0,
        "face_ratio": 0.0,
        "detection_success": 0.0,
        "quality_score": 0.0,
    })
    _session_metrics: list[Dict[str, float]] = field(default_factory=list)

    def _target_image(self, frame: np.ndarray, face_bbox: Optional[BBox]) -> np.ndarray:
        if face_bbox is None:
            return frame
        x1, y1, x2, y2 = face_bbox
        h, w = frame.shape[:2]
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w))
        y2 = max(0, min(int(y2), h))
        if x2 <= x1 or y2 <= y1:
            return frame
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return frame
        return face_crop

    def update(self, frame: np.ndarray, face_bbox: Optional[BBox], face_detected: bool) -> Dict[str, float]:
        """Return current metrics dictionary; expensive metrics run every N frames."""
        self._frame_counter += 1
        fps_live = self.fps_tracker.update()
        detection_success = self.detection_tracker.update(face_detected)

        self._last_metrics["fps"] = _clip_0_100((fps_live / 30.0) * 100.0)
        self._last_metrics["detection_success"] = _clip_0_100(detection_success)
        self._last_metrics["face_ratio"] = face_size_ratio(face_bbox, frame.shape)

        if self._frame_counter % max(1, self.compute_every_n_frames) == 0:
            target = self._target_image(frame, face_bbox)
            self._last_metrics["sharpness"] = sharpness_score(target)
            self._last_metrics["brightness"] = brightness_score(target)
            self._last_metrics["contrast"] = contrast_score(target)
            self._last_metrics["noise"] = noise_score(target)

        self._last_metrics["quality_score"] = camera_quality_score(self._last_metrics)
        self._session_metrics.append(dict(self._last_metrics))
        return dict(self._last_metrics)

    def detection_success_rate(self) -> float:
        """Get current rolling detection success percentage."""
        return float(self._last_metrics.get("detection_success", 0.0))

    def warnings(self, metrics: Dict[str, float]) -> list[str]:
        """Generate user-facing warnings from current metric thresholds."""
        alerts: list[str] = []
        if float(metrics.get("sharpness", 0.0)) < 30.0:
            alerts.append("Blurry frame")
        if float(metrics.get("brightness", 0.0)) < 25.0:
            alerts.append("Low light")
        if float(metrics.get("face_ratio", 0.0)) < 8.0:
            alerts.append("Move closer")
        if float(metrics.get("detection_success", 0.0)) < 60.0:
            alerts.append("Face tracking unstable")
        return alerts

    def session_summary(self, label: str = "camera") -> Dict[str, Any]:
        """Return averaged quality metrics for the current comparison session."""
        if not self._session_metrics:
            return {
                "camera_label": label,
                "samples": 0,
                "averages": dict(self._last_metrics),
            }

        keys = list(self._session_metrics[0].keys())
        averages: Dict[str, float] = {}
        for key in keys:
            averages[key] = float(np.mean([m.get(key, 0.0) for m in self._session_metrics]))

        return {
            "camera_label": label,
            "samples": len(self._session_metrics),
            "averages": averages,
        }
