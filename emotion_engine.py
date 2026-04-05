"""Emotion engine used by the real-time webcam pipeline.

In simple terms, this file is the "brain" behind emotion inference.
It performs these jobs:

1. Model setup and caching:
    - Initializes MediaPipe face detector and FaceMesh (once).
    - Initializes FER (and DeepFace fallback) once and reuses it.

2. Image preparation:
    - Crops and normalizes face images.
    - Applies CLAHE lighting correction so FER is less sensitive to shadows.

3. Emotion scoring:
    - Runs FER on the cropped face.
    - Normalizes probabilities for the supported emotion classes.

4. Intelligence layer for stability:
    - Calibrates and reweights emotion scores.
    - Applies face-quality penalties when frames are low quality.
    - Applies FaceMesh rule corrections (surprise/angry/sad/happy logic).
    - Smooths predictions over time.
    - Uses transition gating + emotion hold to prevent rapid label flicker.

5. Trigger helper logic:
    - Computes simple motion/landmark change signals used by the main loop
      to decide when heavy inference should run.

This file is intentionally split into small helpers so teammates can tune one
part (for example rule thresholds) without touching model loading or smoothing.
"""

from __future__ import annotations

import logging
import math
import os
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple, cast

import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None

try:
    import fer as fer_module
except Exception:  # pragma: no cover
    fer_module = None

try:
    from deepface import DeepFace
except Exception:  # pragma: no cover
    DeepFace = None

from config import (
    CONFIDENCE_CALIBRATION_POWER,
    EMOTION_HOLD_SECONDS,
    EMOTION_WEIGHTS,
    EYE_NARROW_THRESHOLD,
    EYE_OPEN_THRESHOLD,
    ANGRY_MOUTH_OPEN_MAX,
    EYEBROW_DOWN_THRESHOLD,
    EYEBROW_INNER_RAISE_THRESHOLD,
    FACE_CROP_SIZE,
    FACE_PADDING,
    FEATURE_BASELINE_ALPHA,
    HOLD_SWITCH_MARGIN,
    HOLD_ESCAPE_STRONG_HAPPY,
    HOLD_ESCAPE_STRONG_SURPRISE,
    LANDMARK_CHANGE_THRESHOLD,
    LOW_QUALITY_NEUTRAL_BOOST,
    LOW_QUALITY_NON_NEUTRAL_PENALTY,
    MAX_FACES_PROCESS,
    MIN_CONFIDENCE_THRESHOLD,
    MIN_FACE_SIZE,
    MOUTH_OPEN_THRESHOLD,
    LIP_SPREAD_RATIO_THRESHOLD,
    LOW_EXPRESSION_INTENSITY_MAX,
    NEUTRAL_GUARD_BOOST,
    RULE_MIN_ANGRY_SCORE,
    RULE_MIN_HAPPY_SCORE,
    RULE_MIN_SAD_SCORE,
    RULE_MIN_SURPRISE_OR_FEAR_SCORE,
    RAW_OVERRIDE_CONFIDENCE,
    RAW_OVERRIDE_MARGIN,
    RAW_OVERRIDE_STREAK,
    SAD_LIP_SPREAD_MAX,
    SAD_SMILE_MAX,
    SMILE_RATIO_THRESHOLD,
    SMOOTHING_WINDOW_SIZE,
    SURPRISE_BROW_RAISE_THRESHOLD,
    SURPRISE_MOUTH_BOOST_THRESHOLD,
    SUPPORTED_EMOTIONS,
    TRANSITION_MARGIN,
)
from utils import clamp_box

LOGGER = logging.getLogger("emotion_engine")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

_FACE_DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
_FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

_CLAHE_LIGHT = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_CLAHE_CHROMA = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))

_FACE_DETECTOR_CONFIG: Optional[Dict[str, Any]] = None
_FACE_MESH_CONFIG: Optional[Dict[str, Any]] = None
_FER_DETECTOR: Optional[Any] = None


@dataclass
class DetectionCache:
    """Shared cache used by scheduler decisions.

    This is consumed by the realtime loop to avoid running expensive operations
    every frame. It stores the last heavy inference timestamp and key motion
    references (box + landmark signature).
    """

    result: Optional[Dict[str, Any]] = None
    last_detection_time: float = 0.0
    last_detection_frame: int = -1
    last_box: Optional[Tuple[int, int, int, int]] = None
    last_landmark_signature: Optional[Tuple[float, float, float, float]] = None


class EmotionHistory:
    """Legacy rolling history container kept for compatibility with old callers."""

    def __init__(self, window_size: int = SMOOTHING_WINDOW_SIZE) -> None:
        self.window_size = window_size
        self.history: Deque[Dict[str, Any]] = deque(maxlen=window_size)

    def add_prediction(self, emotion: Optional[str], confidence: float, scores: Optional[Dict[str, float]] = None) -> None:
        if not emotion:
            return
        self.history.append({"emotion": emotion, "confidence": float(confidence), "scores": scores or {}})


class EmotionIntelligenceEngine:
    """Stabilizes noisy frame-level predictions into a demo-safe final emotion.

    Key idea: raw FER output can jump between adjacent emotions frame to frame.
    This class introduces calibration, quality-aware weighting, rule corrections,
    temporal smoothing, transition checks, and hold locking.
    """

    def __init__(
        self,
        window_size: int = SMOOTHING_WINDOW_SIZE,
        hold_seconds: float = EMOTION_HOLD_SECONDS,
        transition_margin: float = TRANSITION_MARGIN,
        hold_switch_margin: float = HOLD_SWITCH_MARGIN,
    ) -> None:
        self.window_size = max(5, min(10, int(window_size)))
        self.buffer: Deque[Dict[str, float]] = deque(maxlen=self.window_size)
        self.hold_seconds = float(hold_seconds)
        self.transition_margin = float(transition_margin)
        self.hold_switch_margin = float(hold_switch_margin)
        self.current_emotion: Optional[str] = None
        self.current_confidence: float = 0.0
        self.hold_until: float = 0.0
        self.feature_baseline: Optional[Dict[str, float]] = None
        self.candidate_streak_emotion: Optional[str] = None
        self.candidate_streak_count: int = 0
        self.raw_streak_emotion: Optional[str] = None
        self.raw_streak_count: int = 0

    def evaluate(self, raw_scores: Dict[str, float], features: Dict[str, Any], now_ts: float, face_quality: float = 1.0) -> Dict[str, Any]:
        # Step 1: normalize score confidence shape (soft calibration).
        calibrated = self._calibrate_scores(raw_scores)
        # Step 2: down-weight unstable non-neutral outcomes on poor quality frames.
        quality_adjusted = self._apply_face_quality(calibrated, face_quality)
        # Step 3: maintain per-user geometric baseline for adaptive thresholds.
        baseline = self._update_feature_baseline(features)
        # Step 4: apply class priors to help under-represented emotions.
        weighted = self._apply_weights(quality_adjusted)
        # Step 5: apply deterministic FaceMesh rules.
        corrected_scores, triggers = apply_facial_rules(weighted, features, baseline=baseline)
        # Step 6: apply temporal smoothing across recent frames.
        smoothed_scores = self._smooth(corrected_scores)

        raw_emotion, raw_conf = pick_top(raw_scores)
        # Build candidate from blended maps so smoothing does not fully suppress
        # fresh evidence from the current frame.
        candidate_scores = self._blend_candidate_scores(
            raw_calibrated=quality_adjusted,
            corrected=corrected_scores,
            smoothed=smoothed_scores,
        )

        smoothed_emotion, smoothed_conf = pick_top(smoothed_scores)
        candidate_emotion, candidate_conf = pick_top(candidate_scores)

        # Step 7: prevent sudden transitions unless the new emotion is clearly stronger.
        candidate_emotion, candidate_conf = self._transition_control(candidate_emotion, candidate_conf, smoothed_scores)
        # Step 8: emotion hold lock for visual stability in live demos.
        final_emotion, final_conf = self._apply_hold(candidate_emotion, candidate_conf, now_ts, smoothed_scores)

        # Step 9: if raw FER is repeatedly and strongly confident, allow override.
        final_emotion, final_conf, raw_override = self._raw_confidence_override(
            raw_emotion=raw_emotion,
            raw_conf=raw_conf,
            final_emotion=final_emotion,
            final_conf=final_conf,
        )
        if raw_override:
            triggers.append("raw_confidence_override")

        return {
            "raw_emotion": raw_emotion,
            "raw_confidence": raw_conf,
            "smoothed_emotion": smoothed_emotion,
            "smoothed_confidence": smoothed_conf,
            "final_emotion": final_emotion,
            "final_confidence": final_conf,
            "rule_triggers": triggers,
            "scores": raw_scores,
            "calibrated_scores": calibrated,
            "quality_adjusted_scores": quality_adjusted,
            "weighted_scores": corrected_scores,
            "smoothed_scores": smoothed_scores,
            "candidate_scores": candidate_scores,
            "hold_remaining": max(0.0, self.hold_until - now_ts),
            "face_quality": face_quality,
            "feature_baseline": baseline,
        }

    def _blend_candidate_scores(
        self,
        raw_calibrated: Dict[str, float],
        corrected: Dict[str, float],
        smoothed: Dict[str, float],
    ) -> Dict[str, float]:
        """Blend current-frame and temporal maps to reduce lag-induced errors."""
        merged: Dict[str, float] = {}
        keys = set(raw_calibrated) | set(corrected) | set(smoothed)
        for emotion in keys:
            merged[emotion] = (
                (0.25 * float(raw_calibrated.get(emotion, 0.0)))
                + (0.43 * float(corrected.get(emotion, 0.0)))
                + (0.32 * float(smoothed.get(emotion, 0.0)))
            )
        return normalize_scores(merged)

    def _raw_confidence_override(
        self,
        raw_emotion: Optional[str],
        raw_conf: float,
        final_emotion: str,
        final_conf: float,
    ) -> Tuple[str, float, bool]:
        """Override final output when raw FER is strongly confident and consistent."""
        if raw_emotion is None:
            self.raw_streak_emotion = None
            self.raw_streak_count = 0
            return final_emotion, final_conf, False

        if self.raw_streak_emotion == raw_emotion:
            self.raw_streak_count += 1
        else:
            self.raw_streak_emotion = raw_emotion
            self.raw_streak_count = 1

        if raw_emotion == final_emotion:
            return final_emotion, final_conf, False

        if raw_conf < RAW_OVERRIDE_CONFIDENCE:
            return final_emotion, final_conf, False

        if self.raw_streak_count < RAW_OVERRIDE_STREAK:
            return final_emotion, final_conf, False

        if (raw_conf - final_conf) < RAW_OVERRIDE_MARGIN:
            return final_emotion, final_conf, False

        self.current_emotion = raw_emotion
        self.current_confidence = raw_conf
        return raw_emotion, raw_conf, True

    def _calibrate_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        # Apply a power calibration curve then renormalize.
        calibrated: Dict[str, float] = {}
        for emotion in SUPPORTED_EMOTIONS:
            value = clamp01(scores.get(emotion, 0.0))
            calibrated[emotion] = value ** CONFIDENCE_CALIBRATION_POWER
        return normalize_scores(calibrated)

    def _apply_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        # Emotion priors live in config so behavior can be tuned without code edits.
        weighted: Dict[str, float] = {}
        for emotion, score in scores.items():
            weighted[emotion] = score * float(EMOTION_WEIGHTS.get(emotion, 1.0))
        return normalize_scores(weighted)

    def _apply_face_quality(self, scores: Dict[str, float], quality: float) -> Dict[str, float]:
        # Poor-quality frames (blur, bad exposure) are often overconfident noise.
        adjusted = dict(scores)
        q = clamp01(quality)
        if q < 0.55:
            penalty = 1.0 - ((1.0 - q) * LOW_QUALITY_NON_NEUTRAL_PENALTY)
            for emotion in SUPPORTED_EMOTIONS:
                if emotion == "neutral":
                    continue
                adjusted[emotion] = adjusted.get(emotion, 0.0) * penalty
            # Give neutral a controlled boost so uncertain frames do not oscillate.
            adjusted["neutral"] = adjusted.get("neutral", 0.0) + ((1.0 - q) * LOW_QUALITY_NEUTRAL_BOOST)
        return normalize_scores(adjusted)

    def _update_feature_baseline(self, features: Dict[str, Any]) -> Optional[Dict[str, float]]:
        # No face landmarks available -> keep previous baseline untouched.
        if not features.get("available"):
            return self.feature_baseline

        source = {
            "mouth_open_ratio": float(features.get("mouth_open_ratio", 0.0)),
            "eye_open_ratio": float(features.get("eye_open_ratio", 0.0)),
            "eyebrow_position": float(features.get("eyebrow_position", 0.0)),
            "eyebrow_inner_raise": float(features.get("eyebrow_inner_raise", 0.0)),
            "smile_ratio": float(features.get("smile_ratio", 0.0)),
        }

        if self.feature_baseline is None:
            # First valid observation seeds the adaptive baseline.
            self.feature_baseline = dict(source)
            return self.feature_baseline

        # Exponential moving average keeps baseline stable but adaptive.
        alpha = clamp01(FEATURE_BASELINE_ALPHA)
        for key, value in source.items():
            self.feature_baseline[key] = (alpha * self.feature_baseline[key]) + ((1.0 - alpha) * value)
        return self.feature_baseline

    def _smooth(self, scores: Dict[str, float]) -> Dict[str, float]:
        # Buffer stores only recent corrected scores (window size is configurable).
        if scores:
            self.buffer.append(dict(scores))
        if not self.buffer:
            return {}

        # Use recency weighting so newer frames influence the result more.
        aggregate: Dict[str, float] = defaultdict(float)
        for idx, item in enumerate(self.buffer, start=1):
            weight = idx / len(self.buffer)
            for emotion, score in item.items():
                aggregate[emotion] += float(score) * weight
        return normalize_scores(dict(aggregate))

    def _transition_control(
        self,
        candidate_emotion: Optional[str],
        candidate_conf: float,
        smoothed_scores: Dict[str, float],
    ) -> Tuple[Optional[str], float]:
        if not candidate_emotion:
            return None, 0.0

        if self.current_emotion is None:
            self.candidate_streak_emotion = None
            self.candidate_streak_count = 0
            return candidate_emotion, candidate_conf

        # Keep current emotion unless the candidate exceeds the configured margin.
        current_score = float(smoothed_scores.get(self.current_emotion, self.current_confidence))
        if candidate_emotion == self.current_emotion:
            self.candidate_streak_emotion = None
            self.candidate_streak_count = 0
            return candidate_emotion, candidate_conf

        # Build short evidence streak before allowing frequent flips.
        if self.candidate_streak_emotion == candidate_emotion:
            self.candidate_streak_count += 1
        else:
            self.candidate_streak_emotion = candidate_emotion
            self.candidate_streak_count = 1

        # If candidate repeatedly wins, reduce transition friction.
        if self.candidate_streak_count >= 3 and candidate_conf >= (current_score + (self.transition_margin * 0.5)):
            return candidate_emotion, candidate_conf

        # Very strong candidates can preempt quickly.
        if candidate_conf >= 0.62 and candidate_conf > current_score:
            return candidate_emotion, candidate_conf

        if candidate_emotion != self.current_emotion and (candidate_conf - current_score) < self.transition_margin:
            return self.current_emotion, current_score
        return candidate_emotion, candidate_conf

    def _apply_hold(
        self,
        candidate_emotion: Optional[str],
        candidate_conf: float,
        now_ts: float,
        smoothed_scores: Dict[str, float],
    ) -> Tuple[str, float]:
        if not candidate_emotion:
            if self.current_emotion:
                return self.current_emotion, self.current_confidence
            return "uncertain", 0.0

        if self.current_emotion is None:
            self.current_emotion = candidate_emotion
            self.current_confidence = candidate_conf
            self.hold_until = now_ts + self.hold_seconds
            return self.current_emotion, self.current_confidence

        if candidate_emotion == self.current_emotion:
            # Refresh hold window while emotion is stable.
            self.current_confidence = candidate_conf
            self.hold_until = now_ts + self.hold_seconds
            return self.current_emotion, self.current_confidence

        current_score = float(smoothed_scores.get(self.current_emotion, self.current_confidence))
        # During hold period, require a stronger margin before switching labels.
        if (
            now_ts < self.hold_until
            and candidate_emotion == "happy"
            and candidate_conf >= HOLD_ESCAPE_STRONG_HAPPY
            and candidate_conf > current_score
        ):
            self.current_emotion = candidate_emotion
            self.current_confidence = candidate_conf
            self.hold_until = now_ts + self.hold_seconds
            return self.current_emotion, self.current_confidence

        if (
            now_ts < self.hold_until
            and candidate_emotion == "surprise"
            and candidate_conf >= HOLD_ESCAPE_STRONG_SURPRISE
            and candidate_conf > current_score
        ):
            self.current_emotion = candidate_emotion
            self.current_confidence = candidate_conf
            self.hold_until = now_ts + self.hold_seconds
            return self.current_emotion, self.current_confidence

        if now_ts < self.hold_until and self.candidate_streak_count >= 3 and candidate_conf > current_score:
            self.current_emotion = candidate_emotion
            self.current_confidence = candidate_conf
            self.hold_until = now_ts + self.hold_seconds
            return self.current_emotion, self.current_confidence

        if now_ts < self.hold_until and candidate_conf >= 0.70:
            self.current_emotion = candidate_emotion
            self.current_confidence = candidate_conf
            self.hold_until = now_ts + self.hold_seconds
            return self.current_emotion, self.current_confidence

        if now_ts < self.hold_until and candidate_conf < (current_score + self.hold_switch_margin):
            return self.current_emotion, self.current_confidence

        self.current_emotion = candidate_emotion
        self.current_confidence = candidate_conf
        self.hold_until = now_ts + self.hold_seconds
        return self.current_emotion, self.current_confidence


class _TasksFaceMeshProcessor:
    """Adapter so Tasks FaceLandmarker has the same shape as solutions usage."""

    def __init__(self, mp_module: Any, landmarker: Any) -> None:
        self._mp = mp_module
        self._landmarker = landmarker

    def process(self, frame_bgr: np.ndarray) -> Any:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        return self._landmarker.detect(mp_image)


def _ensure_model(model_path: Path, url: str) -> Path:
    # Model files are downloaded once and reused from local disk.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        return model_path
    urllib.request.urlretrieve(url, str(model_path))
    return model_path


def _init_mediapipe_face_detector() -> Dict[str, Any]:
    # Prefer MediaPipe solutions API when available; fallback to tasks API.
    if mp is None:
        return {"mode": "none", "detector": None}

    solutions = getattr(mp, "solutions", None)
    if solutions is not None and hasattr(solutions, "face_detection"):
        detector = solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.55)
        return {"mode": "solutions", "detector": detector}

    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = Path(__file__).resolve().parent / "models" / "face_detector.tflite"
        model_path = _ensure_model(model_path, _FACE_DETECTOR_MODEL_URL)

        options = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.55,
        )
        detector = vision.FaceDetector.create_from_options(options)
        return {"mode": "tasks", "detector": detector}
    except Exception as exc:
        LOGGER.warning("MediaPipe face detector init failed: %s", exc)
        return {"mode": "none", "detector": None}


def _init_mediapipe_facemesh() -> Dict[str, Any]:
    # Same strategy as face detector: solutions first, tasks fallback.
    if mp is None:
        return {"mode": "none", "processor": None}

    solutions = getattr(mp, "solutions", None)
    if solutions is not None and hasattr(solutions, "face_mesh"):
        face_mesh = solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        )
        return {"mode": "solutions", "face_mesh": face_mesh}

    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
        model_path = _ensure_model(model_path, _FACE_LANDMARKER_MODEL_URL)

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
        return {"mode": "tasks", "processor": _TasksFaceMeshProcessor(mp, landmarker)}
    except Exception as exc:
        LOGGER.warning("MediaPipe FaceMesh init failed: %s", exc)
        return {"mode": "none", "processor": None}


def get_face_detector_config() -> Dict[str, Any]:
    # Lazy singleton creation avoids startup overhead for import-only workflows.
    global _FACE_DETECTOR_CONFIG
    if _FACE_DETECTOR_CONFIG is None:
        _FACE_DETECTOR_CONFIG = _init_mediapipe_face_detector()
    return _FACE_DETECTOR_CONFIG


def get_face_mesh_config() -> Dict[str, Any]:
    # Lazy singleton creation avoids repeated graph construction.
    global _FACE_MESH_CONFIG
    if _FACE_MESH_CONFIG is None:
        _FACE_MESH_CONFIG = _init_mediapipe_facemesh()
    return _FACE_MESH_CONFIG


def _get_fer_detector() -> Optional[Any]:
    # FER object creation is expensive; initialize once and cache.
    global _FER_DETECTOR
    if _FER_DETECTOR is not None:
        return _FER_DETECTOR
    if fer_module is None:
        return None
    try:
        fer_class = getattr(fer_module, "FER", None)
        if fer_class is None:
            return None
        _FER_DETECTOR = fer_class(mtcnn=False)
        return _FER_DETECTOR
    except Exception as exc:
        LOGGER.warning("FER init failed: %s", exc)
        return None


def _frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    # MediaPipe and FER expect RGB, while OpenCV camera frames are BGR.
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    # Ensure scores are a probability distribution and include all classes.
    total = float(sum(max(0.0, v) for v in scores.values()))
    if total <= 1e-8:
        return {emotion: 0.0 for emotion in SUPPORTED_EMOTIONS}
    normalized = {emotion: max(0.0, float(value)) / total for emotion, value in scores.items()}
    for emotion in SUPPORTED_EMOTIONS:
        normalized.setdefault(emotion, 0.0)
    return normalized


def pick_top(scores: Dict[str, float]) -> Tuple[Optional[str], float]:
    if not scores:
        return None, 0.0
    emotion, confidence = max(scores.items(), key=lambda item: item[1])
    return emotion, float(confidence)


def _point(landmarks: List[Any], index: int) -> Tuple[float, float]:
    item = landmarks[index]
    return float(item.x), float(item.y)


def _distance(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def compute_face_quality(face_crop: np.ndarray) -> float:
    """Estimate face quality from sharpness and brightness for confidence modulation."""
    if face_crop is None or face_crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))

    # Sharpness from Laplacian variance; brightness favors mid-range exposure.
    sharpness_score = clamp01(blur_var / 220.0)
    brightness_score = clamp01(1.0 - (abs(brightness - 122.0) / 122.0))
    return clamp01((0.6 * sharpness_score) + (0.4 * brightness_score))


def _relative_bbox_to_xyxy(relative_box: Any, frame_w: int, frame_h: int) -> Optional[Tuple[int, int, int, int]]:
    if relative_box is None:
        return None

    if hasattr(relative_box, "xmin"):
        x1 = int(relative_box.xmin * frame_w)
        y1 = int(relative_box.ymin * frame_h)
        x2 = x1 + int(relative_box.width * frame_w)
        y2 = y1 + int(relative_box.height * frame_h)
        return clamp_box(x1, y1, x2, y2, frame_w, frame_h)

    if hasattr(relative_box, "origin_x"):
        x1 = int(relative_box.origin_x)
        y1 = int(relative_box.origin_y)
        x2 = x1 + int(relative_box.width)
        y2 = y1 + int(relative_box.height)
        return clamp_box(x1, y1, x2, y2, frame_w, frame_h)

    return None


def preprocess_face(face_crop: np.ndarray) -> Optional[np.ndarray]:
    """Normalize lighting using CLAHE in LAB and YCrCb, then resize to 128x128."""
    if face_crop is None or face_crop.size == 0:
        return None

    resized = cv2.resize(face_crop, FACE_CROP_SIZE, interpolation=cv2.INTER_AREA)

    # First pass: CLAHE on LAB luminance channel.
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = _CLAHE_LIGHT.apply(l_channel)
    lab_corrected = cv2.merge((l_channel, a_channel, b_channel))
    after_lab = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

    # Second pass: CLAHE on Y channel in YCrCb space.
    ycrcb = cv2.cvtColor(after_lab, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
    y_channel = _CLAHE_CHROMA.apply(y_channel)
    ycrcb_corrected = cv2.merge((y_channel, cr_channel, cb_channel))
    return cv2.cvtColor(ycrcb_corrected, cv2.COLOR_YCrCb2BGR)


def detect_faces_mediapipe(frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
    """Detect faces with MediaPipe and keep only strongest faces."""
    if frame is None or frame.size == 0:
        return []

    detector_config = get_face_detector_config()
    detector = detector_config.get("detector")
    mode = detector_config.get("mode")
    if detector is None or mode == "none":
        return []

    frame_h, frame_w = frame.shape[:2]
    rgb = _frame_to_rgb(frame)

    if mode == "solutions":
        result = detector.process(rgb)
        detections = result.detections if result and result.detections else []
    else:
        mp_module = cast(Any, mp)
        mp_image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        detections = result.detections if result and result.detections else []

    # Keep only the highest-confidence face for low latency and stable behavior.
    boxes: List[Tuple[int, int, int, int, float]] = []
    for det in detections[:MAX_FACES_PROCESS]:
        location_data = getattr(det, "location_data", None)
        relative_box = getattr(location_data, "relative_bounding_box", None) if location_data else None
        if relative_box is None:
            relative_box = getattr(det, "bounding_box", None)

        box = _relative_bbox_to_xyxy(relative_box, frame_w, frame_h)
        if box is None:
            continue

        x1, y1, x2, y2 = box
        if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE:
            continue

        score = 0.0
        raw_score = getattr(det, "score", None)
        if raw_score:
            score = float(raw_score[0])
        else:
            categories = getattr(det, "categories", None)
            if categories:
                score = float(getattr(categories[0], "score", 0.0))

        boxes.append((x1, y1, x2, y2, score))

    boxes.sort(key=lambda item: item[4], reverse=True)
    return boxes[:1]


def extract_face(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    face_size: Tuple[int, int] = FACE_CROP_SIZE,
    padding: int = FACE_PADDING,
) -> Optional[np.ndarray]:
    """Crop one face from the original frame with padding."""
    if frame is None or frame.size == 0:
        return None

    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = box
    padded = clamp_box(x1 - padding, y1 - padding, x2 + padding, y2 + padding, frame_w, frame_h)
    if padded is None:
        return None

    px1, py1, px2, py2 = padded
    crop = frame[py1:py2, px1:px2]
    if crop.size == 0:
        return None

    return cv2.resize(crop, face_size, interpolation=cv2.INTER_AREA)


crop_face = extract_face


def detect_emotion_fer(face_crop: np.ndarray) -> Tuple[Optional[str], Dict[str, float]]:
    """Run FER on a cropped face and return normalized probabilities."""
    if face_crop is None or face_crop.size == 0:
        return None, {}

    # Primary path: FER package (fast and straightforward for emotion logits).
    fer_model = _get_fer_detector()
    if fer_model is not None:
        try:
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            detected = fer_model.detect_emotions(rgb)
            if detected:
                emotions = cast(Dict[str, float], detected[0].get("emotions", {}))
                scores = {name: clamp01(float(value)) for name, value in emotions.items()}
                normalized = normalize_scores(scores)
                return pick_top(normalized)[0], normalized
        except Exception:
            pass

    # Fallback path: DeepFace emotion analysis when FER is unavailable.
    if DeepFace is None:
        return None, {}

    try:
        result = DeepFace.analyze(face_crop, actions=["emotion"], enforce_detection=False, silent=True)
        if isinstance(result, list):
            result = result[0] if result else {}

        emotion_dict = cast(Dict[str, float], result.get("emotion", {})) if isinstance(result, dict) else {}
        scores: Dict[str, float] = {}
        for name, raw in emotion_dict.items():
            score = float(raw)
            if score > 1.0:
                score /= 100.0
            scores[name] = clamp01(score)
        normalized = normalize_scores(scores)
        return pick_top(normalized)[0], normalized
    except Exception:
        return None, {}


analyze_emotion_fer = detect_emotion_fer


def analyze_emotion(face_crop: np.ndarray) -> Tuple[Optional[str], float, Dict[str, float]]:
    emotion, scores = detect_emotion_fer(face_crop)
    confidence = float(scores.get(emotion, 0.0)) if emotion else 0.0
    return emotion, confidence, scores


def _face_landmarks(frame_bgr: np.ndarray, mediapipe_config: Optional[Dict[str, Any]]) -> Optional[List[Any]]:
    # Returns first face landmarks only (single-face pipeline).
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    config = mediapipe_config or get_face_mesh_config()
    mode = config.get("mode")
    if mode == "none":
        return None

    if mode == "solutions":
        face_mesh = config.get("face_mesh")
        if face_mesh is None:
            return None
        rgb = _frame_to_rgb(frame_bgr)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        return list(results.multi_face_landmarks[0].landmark)

    processor = config.get("processor")
    if processor is None:
        return None

    results = processor.process(frame_bgr)
    if not results or not getattr(results, "face_landmarks", None):
        return None

    return list(results.face_landmarks[0])


def extract_features_mediapipe(frame: np.ndarray, mediapipe_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extract facial geometry features for rule-based corrections."""
    features: Dict[str, Any] = {
        "mouth_open_ratio": 0.0,
        "lip_spread_ratio": 0.0,
        "eye_open_ratio": 0.0,
        "eyebrow_position": 0.0,
        "eyebrow_inner_raise": 0.0,
        "smile_ratio": 0.0,
        "expression_intensity": 0.0,
        "available": False,
    }

    landmarks = _face_landmarks(frame, mediapipe_config)
    if not landmarks:
        return features

    # On any extraction failure, return safe defaults so pipeline remains stable.
    try:
        # Face width is used as a normalization factor for scale invariance.
        face_width = max(_distance(_point(landmarks, 234), _point(landmarks, 454)), 1e-6)

        mouth_open = _distance(_point(landmarks, 13), _point(landmarks, 14))
        mouth_width = max(_distance(_point(landmarks, 61), _point(landmarks, 291)), 1e-6)
        mouth_open_ratio = mouth_open / mouth_width
        lip_spread_ratio = mouth_width / face_width

        left_eye_open = _distance(_point(landmarks, 159), _point(landmarks, 145))
        right_eye_open = _distance(_point(landmarks, 386), _point(landmarks, 374))
        eye_open_ratio = ((left_eye_open + right_eye_open) * 0.5) / face_width

        left_brow_to_eye = _distance(_point(landmarks, 70), _point(landmarks, 159))
        right_brow_to_eye = _distance(_point(landmarks, 300), _point(landmarks, 386))
        eyebrow_position = ((left_brow_to_eye + right_brow_to_eye) * 0.5) / face_width

        left_inner = _distance(_point(landmarks, 107), _point(landmarks, 159))
        right_inner = _distance(_point(landmarks, 336), _point(landmarks, 386))
        eyebrow_inner_raise = ((left_inner + right_inner) * 0.5) / face_width

        mouth_center_y = (_point(landmarks, 13)[1] + _point(landmarks, 14)[1]) * 0.5
        corner_y = (_point(landmarks, 61)[1] + _point(landmarks, 291)[1]) * 0.5
        smile_ratio = (mouth_center_y - corner_y)

        expression_intensity = (
            abs(mouth_open_ratio - float(MOUTH_OPEN_THRESHOLD))
            + abs(eye_open_ratio - float(EYE_OPEN_THRESHOLD))
            + abs(eyebrow_inner_raise - float(EYEBROW_INNER_RAISE_THRESHOLD))
            + abs(smile_ratio - float(SMILE_RATIO_THRESHOLD))
        ) * 0.25

        features.update(
            {
                "mouth_open_ratio": float(mouth_open_ratio),
                "lip_spread_ratio": float(lip_spread_ratio),
                "eye_open_ratio": float(eye_open_ratio),
                "eyebrow_position": float(eyebrow_position),
                "eyebrow_inner_raise": float(eyebrow_inner_raise),
                "smile_ratio": float(smile_ratio),
                "expression_intensity": float(expression_intensity),
                "available": True,
            }
        )
    except Exception:
        return features

    return features


def landmark_signature(features: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    # Compact signature used for quick "landmarks changed" trigger decisions.
    if not features.get("available"):
        return None
    return (
        float(features.get("mouth_open_ratio", 0.0)),
        float(features.get("eye_open_ratio", 0.0)),
        float(features.get("eyebrow_position", 0.0)),
        float(features.get("smile_ratio", 0.0)),
    )


def landmark_changed(
    previous: Optional[Tuple[float, float, float, float]],
    current: Optional[Tuple[float, float, float, float]],
    threshold: float = LANDMARK_CHANGE_THRESHOLD,
) -> bool:
    # L1 distance is enough here and cheaper than more complex metrics.
    if previous is None or current is None:
        return False
    deltas = [abs(a - b) for a, b in zip(previous, current)]
    delta = sum(deltas)
    max_delta = max(deltas) if deltas else 0.0
    return delta >= threshold and max_delta >= (threshold * 0.35)


def face_movement_detected(
    previous_box: Optional[Tuple[int, int, int, int]],
    current_box: Optional[Tuple[int, int, int, int]],
    movement_threshold_px: float,
) -> bool:
    # Compare center-point motion between previous and current face boxes.
    if previous_box is None or current_box is None:
        return False

    prev_cx = (previous_box[0] + previous_box[2]) * 0.5
    prev_cy = (previous_box[1] + previous_box[3]) * 0.5
    curr_cx = (current_box[0] + current_box[2]) * 0.5
    curr_cy = (current_box[1] + current_box[3]) * 0.5
    return math.hypot(curr_cx - prev_cx, curr_cy - prev_cy) >= movement_threshold_px


def apply_facial_rules(
    scores: Dict[str, float],
    features: Dict[str, Any],
    baseline: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], List[str]]:
    """Apply FaceMesh logic rules and return updated scores with trigger list."""
    if not scores:
        return {}, []

    adjusted = dict(scores)
    triggers: List[str] = []
    neutral_guard_active = False

    baseline = baseline or {}
    mouth_val = float(features.get("mouth_open_ratio", 0.0))
    lip_spread_val = float(features.get("lip_spread_ratio", 0.0))
    eye_val = float(features.get("eye_open_ratio", 0.0))
    eyebrow_val = float(features.get("eyebrow_position", 0.0))
    inner_brow_val = float(features.get("eyebrow_inner_raise", 0.0))
    smile_val = float(features.get("smile_ratio", 0.0))
    expression_intensity = float(features.get("expression_intensity", 0.0))

    # Baseline-aware thresholds make rules personalized and less jittery.
    mouth_threshold = max(MOUTH_OPEN_THRESHOLD, float(baseline.get("mouth_open_ratio", 0.0)) * 1.45)
    eye_open_threshold = max(EYE_OPEN_THRESHOLD, float(baseline.get("eye_open_ratio", 0.0)) * 1.20)
    eye_narrow_threshold = min(EYE_NARROW_THRESHOLD, float(baseline.get("eye_open_ratio", 0.0)) * 0.82)
    eyebrow_down_threshold = min(EYEBROW_DOWN_THRESHOLD, float(baseline.get("eyebrow_position", 0.0)) * 0.88)
    inner_raise_threshold = max(EYEBROW_INNER_RAISE_THRESHOLD, float(baseline.get("eyebrow_inner_raise", 0.0)) * 1.12)
    smile_threshold = max(SMILE_RATIO_THRESHOLD, float(baseline.get("smile_ratio", 0.0)) + 0.015)

    mouth_open = mouth_val >= mouth_threshold
    eyes_open = eye_val >= eye_open_threshold
    eyes_narrow = eye_val <= eye_narrow_threshold
    eyebrows_down = eyebrow_val <= eyebrow_down_threshold
    inner_raised = inner_brow_val >= inner_raise_threshold
    smile_high = smile_val >= smile_threshold
    lips_spread = lip_spread_val >= LIP_SPREAD_RATIO_THRESHOLD

    # Guard neutral when expression cues are weak so neutral is not misread as sad/angry.
    neutral_support = adjusted.get("neutral", 0.0)
    if expression_intensity <= LOW_EXPRESSION_INTENSITY_MAX and neutral_support >= 0.35:
        adjusted["neutral"] = adjusted.get("neutral", 0.0) + NEUTRAL_GUARD_BOOST
        adjusted["sad"] = adjusted.get("sad", 0.0) * 0.86
        adjusted["angry"] = adjusted.get("angry", 0.0) * 0.90
        triggers.append("neutral_guard_low_expression")
        neutral_guard_active = True

    surprise_prior = adjusted.get("surprise", 0.0)
    fear_prior = adjusted.get("fear", 0.0)
    angry_prior = adjusted.get("angry", 0.0)
    sad_prior = adjusted.get("sad", 0.0)

    # Surprise heuristic: open mouth + raised inner brows should be enough on its own
    # when the geometry is strong, with eyes-open acting as a confidence booster.
    if mouth_open and inner_raised:
        surprise_boost = 0.26
        if eyes_open:
            surprise_boost += 0.14
        if expression_intensity >= (LOW_EXPRESSION_INTENSITY_MAX + 0.04):
            surprise_boost += 0.10
        if surprise_prior >= RULE_MIN_SURPRISE_OR_FEAR_SCORE or fear_prior >= RULE_MIN_SURPRISE_OR_FEAR_SCORE:
            surprise_boost += 0.08
        adjusted["surprise"] = max(adjusted.get("surprise", 0.0) + surprise_boost, fear_prior * 0.92)
        adjusted["angry"] = adjusted.get("angry", 0.0) * 0.92
        adjusted["sad"] = adjusted.get("sad", 0.0) * 0.95
        triggers.append("surprise_boost_mouth_open_brow_raised")

    # Secondary surprise cue: mouth open + eyes open, but only when surprise/fear is plausible.
    if mouth_open and eyes_open and max(surprise_prior, fear_prior) >= RULE_MIN_SURPRISE_OR_FEAR_SCORE:
        adjusted["surprise"] = adjusted.get("surprise", 0.0) + 0.18
        triggers.append("surprise_boost_mouth_open_eyes_open")

    # Angry heuristic: lowered brows + narrowed eyes with stronger angry support than sad.
    if (
        eyebrows_down
        and eyes_narrow
        and mouth_val <= ANGRY_MOUTH_OPEN_MAX
        and adjusted.get("angry", 0.0) >= RULE_MIN_ANGRY_SCORE
        and angry_prior >= (sad_prior + 0.04)
    ):
        adjusted["angry"] = adjusted.get("angry", 0.0) + 0.28
        adjusted["sad"] = adjusted.get("sad", 0.0) * 0.90
        triggers.append("angry_boost_eyebrows_down_eyes_narrow")

    # Sad heuristic: inner eyebrows raised + low eye openness with sad support above angry.
    sad_face_shape = (smile_val <= SAD_SMILE_MAX) and (lip_spread_val <= SAD_LIP_SPREAD_MAX)
    if (
        (not neutral_guard_active)
        and inner_raised
        and eyes_narrow
        and sad_face_shape
        and adjusted.get("sad", 0.0) >= RULE_MIN_SAD_SCORE
        and sad_prior >= (angry_prior + 0.04)
    ):
        adjusted["sad"] = adjusted.get("sad", 0.0) + 0.24
        adjusted["angry"] = adjusted.get("angry", 0.0) * 0.90
        triggers.append("sad_boost_inner_brows_raised_low_eyes")

    # Happy override only when happy already has enough baseline support.
    if (smile_high or lips_spread) and adjusted.get("happy", 0.0) >= RULE_MIN_HAPPY_SCORE:
        adjusted["happy"] = max(adjusted.get("happy", 0.0) + 0.35, adjusted.get("surprise", 0.0))
        triggers.append("happy_override_smile_or_lipspread")

    if baseline:
        triggers.append("adaptive_baseline_applied")

    return normalize_scores(adjusted), triggers


def apply_emotion_rules(emotion: Optional[str], features: Dict[str, Any], enabled: bool = True) -> Optional[str]:
    """Backward-compatible single-label rule helper."""
    if not enabled or not emotion:
        return emotion
    baseline = {name: 0.0 for name in SUPPORTED_EMOTIONS}
    baseline[emotion] = 1.0
    adjusted, _ = apply_facial_rules(baseline, features)
    return pick_top(adjusted)[0]


def smooth_emotions(
    history: Iterable[Dict[str, Any]],
    window_size: int = SMOOTHING_WINDOW_SIZE,
    min_confidence: float = MIN_CONFIDENCE_THRESHOLD,
) -> Tuple[Optional[str], float]:
    # Compatibility smoother used by older modules still passing history deques.
    recent = list(history)[-window_size:]
    if not recent:
        return None, 0.0

    aggregated: Dict[str, float] = defaultdict(float)
    for index, item in enumerate(recent, start=1):
        scores = cast(Dict[str, float], item.get("scores", {}))
        if not scores:
            emotion = item.get("emotion") or item.get("final_emotion")
            confidence = float(item.get("confidence", item.get("final_confidence", 0.0)))
            if emotion and confidence >= min_confidence:
                aggregated[str(emotion)] += confidence * (index / len(recent))
            continue

        weight = index / len(recent)
        for emotion, score in scores.items():
            aggregated[str(emotion)] += float(score) * weight

    if not aggregated:
        return None, 0.0

    top_emotion, top_conf = max(aggregated.items(), key=lambda item: item[1])
    return top_emotion, clamp01(top_conf)


def get_final_emotion(
    raw_scores: Dict[str, float],
    history: Deque[Dict[str, Any]],
    features: Dict[str, Any],
    enable_rules: bool = True,
    confidence_threshold: float = MIN_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """Backward-compatible helper for modules that still use history deque."""
    raw_emotion, raw_confidence = pick_top(raw_scores)
    if raw_emotion is None:
        return {
            "raw_emotion": None,
            "raw_confidence": 0.0,
            "corrected_emotion": None,
            "smoothed_emotion": None,
            "smoothed_confidence": 0.0,
            "final_emotion": "uncertain",
            "final_confidence": 0.0,
            "scores": {},
        }

    # Keep old behavior shape so legacy callers continue to work unchanged.
    calibrated = normalize_scores(raw_scores)
    corrected_scores, triggers = apply_facial_rules(calibrated, features) if enable_rules else (calibrated, [])
    corrected_emotion, corrected_conf = pick_top(corrected_scores)

    history.append(
        {
            "emotion": corrected_emotion,
            "confidence": corrected_conf,
            "scores": corrected_scores,
            "raw_emotion": raw_emotion,
            "raw_confidence": raw_confidence,
            "rule_triggers": triggers,
        }
    )

    smoothed_emotion, smoothed_conf = smooth_emotions(history)
    final_emotion = smoothed_emotion or corrected_emotion or raw_emotion
    final_conf = smoothed_conf if smoothed_emotion else corrected_conf

    if final_conf < confidence_threshold:
        final_emotion = "uncertain"

    return {
        "raw_emotion": raw_emotion,
        "raw_confidence": raw_confidence,
        "corrected_emotion": corrected_emotion,
        "smoothed_emotion": smoothed_emotion,
        "smoothed_confidence": smoothed_conf,
        "final_emotion": final_emotion,
        "final_confidence": final_conf,
        "scores": corrected_scores,
        "rule_triggers": triggers,
    }


def should_run_detection(
    current_time: float,
    cache: DetectionCache,
    current_frame_index: int,
    time_interval_seconds: float,
    face_moved: bool = False,
    landmarks_changed: bool = False,
) -> bool:
    """Schedule heavy detection on time, movement, or landmark-change trigger."""
    # This helper is intentionally simple because scheduling policy is handled in
    # realtime_emotion.py and this function is used in tight per-frame loops.
    if cache.result is None or cache.last_detection_frame < 0:
        return True

    if face_moved or landmarks_changed:
        return True

    if (current_time - cache.last_detection_time) >= float(time_interval_seconds):
        return True

    return False


def cache_last_result(
    cache: DetectionCache,
    result: Dict[str, Any],
    current_time: float,
    current_frame_index: int,
    current_box: Optional[Tuple[int, int, int, int]] = None,
    current_landmark_signature: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, Any]:
    # Persist references needed for next-frame scheduling decisions.
    cache.result = dict(result)
    cache.last_detection_time = float(current_time)
    cache.last_detection_frame = int(current_frame_index)
    cache.last_box = current_box
    cache.last_landmark_signature = current_landmark_signature
    return cache.result


def log_rate_limited_warning(message: str, last_error_log_ts: float, min_interval: float = 5.0) -> float:
    # Avoid spamming logs for repeated transient failures.
    now = float(cv2.getTickCount()) / max(float(cv2.getTickFrequency()), 1.0)
    if (now - last_error_log_ts) >= min_interval:
        LOGGER.warning(message)
        return now
    return last_error_log_ts


def print_score_comparison(
    track_id: int,
    deepface_scores: Dict[str, float],
    fer_scores: Dict[str, float],
    deepface_emotion: Optional[str],
    fer_emotion: Optional[str],
) -> None:
    LOGGER.info(
        "Track %s: DeepFace=%s (%.2f) | FER=%s (%.2f)",
        track_id,
        deepface_emotion,
        deepface_scores.get(deepface_emotion or "", 0.0),
        fer_emotion,
        fer_scores.get(fer_emotion or "", 0.0),
    )


def fuse_emotion_scores(
    primary_scores: Dict[str, float],
    secondary_scores: Dict[str, float],
    primary_weight: float = 0.65,
    secondary_weight: float = 0.35,
) -> Dict[str, float]:
    # Utility for optional multi-model fusion in compatibility paths.
    fused: Dict[str, float] = {}
    for emotion in set(primary_scores) | set(secondary_scores):
        p = float(primary_scores.get(emotion, 0.0))
        s = float(secondary_scores.get(emotion, 0.0))
        fused[emotion] = (p * primary_weight) + (s * secondary_weight)
    return normalize_scores(fused)


def decide_emotion_from_scores(scores: Dict[str, float], threshold: float = MIN_CONFIDENCE_THRESHOLD) -> Tuple[Optional[str], float, Dict[str, float]]:
    emotion, confidence = pick_top(scores)
    if emotion is None or confidence < threshold:
        return None, float(confidence), scores
    return emotion, float(confidence), scores


__all__ = [
    "DetectionCache",
    "EmotionHistory",
    "EmotionIntelligenceEngine",
    "analyze_emotion",
    "analyze_emotion_fer",
    "apply_emotion_rules",
    "cache_last_result",
    "crop_face",
    "compute_face_quality",
    "decide_emotion_from_scores",
    "detect_emotion_fer",
    "detect_faces_mediapipe",
    "extract_face",
    "extract_features_mediapipe",
    "face_movement_detected",
    "fuse_emotion_scores",
    "get_face_detector_config",
    "get_face_mesh_config",
    "get_final_emotion",
    "landmark_changed",
    "landmark_signature",
    "log_rate_limited_warning",
    "preprocess_face",
    "print_score_comparison",
    "should_run_detection",
    "smooth_emotions",
]
