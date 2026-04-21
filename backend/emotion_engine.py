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
import json
import time
import urllib.request
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
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
    EMOTION_LOG_PATH,
    LOG_LEVEL,
    CONFIDENCE_CALIBRATION_POWER,
    EMA_ALPHA,
    EMOTION_HOLD_SECONDS,
    EMOTION_WEIGHTS,
    FINAL_HOLD_SECONDS,
    FORCE_BREAK_CONFIDENCE_THRESHOLD,
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
    HIGH_CONFIDENCE_OVERRIDE_THRESHOLD,
    HOLD_BREAK_CONFIDENCE_DELTA,
    IOT_ACTUATION_ENABLED,
    IOT_ACTUATION_HOLD_SECONDS,
    IOT_BLOCK_UNCERTAIN,
    IOT_CONFIRM_STREAK,
    IOT_MIN_CONFIDENCE,
    IOT_OVERRIDE_MARGIN,
    LANDMARK_CHANGE_THRESHOLD,
    LOW_QUALITY_NEUTRAL_BOOST,
    LOW_QUALITY_NON_NEUTRAL_PENALTY,
    LOW_CONFIDENCE_GAP_THRESHOLD,
    MAX_FACES_PROCESS,
    MIN_CONFIDENCE_THRESHOLD,
    MIN_FACE_SIZE,
    MOUTH_OPEN_THRESHOLD,
    LIP_SPREAD_RATIO_THRESHOLD,
    GEOMETRY_APPLY_RAW_MAX_CONFIDENCE,
    GEOMETRY_OVERRIDE_MIN_STRENGTH,
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
    STABLE_CHANGE_CONFIDENCE_THRESHOLD,
    SMOOTHING_WINDOW_SIZE,
    SURPRISE_BROW_RAISE_THRESHOLD,
    SURPRISE_MOUTH_BOOST_THRESHOLD,
    SUPPORTED_EMOTIONS,
    TEMPORAL_WINDOW_FPS,
    TEMPORAL_WINDOW_SECONDS,
    TEMPORAL_STRONG_EMOTION_THRESHOLD,
    TEMPORAL_CONSENSUS_MIN_FRAMES,
    TRANSITION_MARGIN,
    TRANSITION_COOLDOWN_SECONDS,
    HOLD_OVERRIDE_MARGIN,
    USE_EMA_SMOOTHING,
    WEAK_SIGNAL_CONFIDENCE_THRESHOLD,
)
from utils import clamp_box

LOGGER = logging.getLogger("emotion_engine")
if not LOGGER.handlers:
    _configured_level = getattr(logging, str(LOG_LEVEL).upper(), logging.WARNING)
    logging.basicConfig(level=_configured_level, format="[%(levelname)s] %(message)s")


def log_emotion_frame(data: dict) -> None:
    """Append one structured frame record to the backend emotion JSONL log."""
    payload = dict(data)
    payload["timestamp"] = time.time()
    try:
        log_path = Path(EMOTION_LOG_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        # Logging must never interrupt realtime inference.
        pass

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


@dataclass
class EmotionEngineConfig:
    """Configuration bundle for the intelligence layer."""

    supported_emotions: Tuple[str, ...] = tuple(SUPPORTED_EMOTIONS)
    confidence_calibration_power: float = CONFIDENCE_CALIBRATION_POWER
    emotion_weights: Dict[str, float] = field(default_factory=lambda: dict(EMOTION_WEIGHTS))

    min_confidence_threshold: float = MIN_CONFIDENCE_THRESHOLD
    confidence_gap_threshold: float = LOW_CONFIDENCE_GAP_THRESHOLD

    smoothing_window_size: int = SMOOTHING_WINDOW_SIZE
    use_ema_smoothing: bool = USE_EMA_SMOOTHING
    ema_alpha: float = EMA_ALPHA

    transition_margin: float = TRANSITION_MARGIN
    transition_cooldown_seconds: float = TRANSITION_COOLDOWN_SECONDS

    hold_seconds: float = EMOTION_HOLD_SECONDS
    hold_switch_margin: float = HOLD_SWITCH_MARGIN
    hold_override_margin: float = HOLD_OVERRIDE_MARGIN

    raw_override_confidence: float = RAW_OVERRIDE_CONFIDENCE
    raw_override_margin: float = RAW_OVERRIDE_MARGIN
    raw_override_streak: int = RAW_OVERRIDE_STREAK

    iot_actuation_enabled: bool = IOT_ACTUATION_ENABLED
    iot_actuation_hold_seconds: float = IOT_ACTUATION_HOLD_SECONDS
    iot_min_confidence: float = IOT_MIN_CONFIDENCE
    iot_confirm_streak: int = IOT_CONFIRM_STREAK
    iot_override_margin: float = IOT_OVERRIDE_MARGIN
    iot_block_uncertain: bool = IOT_BLOCK_UNCERTAIN


@dataclass
class LowConfidenceResult:
    filtered_scores: Dict[str, float]
    status: str
    reason: str
    uncertain: bool


@dataclass
class ProcessEmotionResult:
    raw_scores: Dict[str, float]
    calibrated_scores: Dict[str, float]
    weighted_scores: Dict[str, float]
    rule_adjusted_scores: Dict[str, float]
    smoothed_scores: Dict[str, float]

    raw_top_emotion: Optional[str]
    raw_top_confidence: float
    stable_emotion: Optional[str]
    stable_confidence: float
    smoothed_top_emotion: Optional[str]
    smoothed_top_confidence: float
    decision_source: str
    geometry_emotion: Optional[str]
    geometry_strength: float
    geometry_reason: str
    expression_intensity: float

    final_emotion: str
    final_confidence: float
    iot_emotion: str
    iot_hold_remaining: float
    iot_transition_decision: str

    triggered_rules: List[str]
    hold_active: bool
    hold_remaining: float
    transition_decision: str

    uncertain: bool
    status: str
    debug_reason: str

    low_confidence_reason: str
    filtered_scores: Dict[str, float]


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
        config: Optional[EmotionEngineConfig] = None,
    ) -> None:
        self.config = config or EmotionEngineConfig()
        self.window_size = max(5, min(10, int(window_size or self.config.smoothing_window_size)))
        self.buffer: Deque[Dict[str, float]] = deque(maxlen=self.window_size)
        self.hold_seconds = float(hold_seconds or self.config.hold_seconds)
        self.transition_margin = float(transition_margin or self.config.transition_margin)
        self.hold_switch_margin = float(hold_switch_margin or self.config.hold_switch_margin)

        self.current_emotion: Optional[str] = None
        self.current_confidence: float = 0.0
        self.hold_until: float = 0.0

        self.feature_baseline: Optional[Dict[str, float]] = None

        self.candidate_streak_emotion: Optional[str] = None
        self.candidate_streak_count: int = 0

        self.raw_streak_emotion: Optional[str] = None
        self.raw_streak_count: int = 0

        self.last_transition_ts: float = 0.0
        self.ema_scores: Optional[Dict[str, float]] = None

        self.iot_emotion: str = "neutral"
        self.iot_confidence: float = 0.0
        self.iot_hold_until: float = 0.0
        self.iot_candidate_emotion: Optional[str] = None
        self.iot_candidate_count: int = 0

        temporal_frames = max(2, int(TEMPORAL_WINDOW_FPS * TEMPORAL_WINDOW_SECONDS))
        self.temporal_window: Deque[Dict[str, Any]] = deque(maxlen=temporal_frames)
        self.last_emotion: Optional[str] = None
        self.final_emotion_state: str = "uncertain"
        self.last_change_time: float = 0.0
        self.last_final_emotion: Optional[str] = None
        self.repeat_count: int = 0
        self.iot_last_change: float = 0.0

        self._log_frame_index: int = 0

    def process_emotion_signal(
        self,
        fer_scores: Dict[str, float],
        landmarks: Dict[str, Any],
        current_state: Optional[Dict[str, Any]],
        timestamp: float,
        movement_metrics: Optional[Dict[str, Any]] = None,
        face_quality: float = 1.0,
    ) -> ProcessEmotionResult:
        """Run full intelligence pipeline and return rich debug-friendly output."""
        if not fer_scores:
            return ProcessEmotionResult(
                raw_scores={},
                calibrated_scores={},
                weighted_scores={},
                rule_adjusted_scores={},
                smoothed_scores={},
                raw_top_emotion=None,
                raw_top_confidence=0.0,
                stable_emotion=None,
                stable_confidence=0.0,
                smoothed_top_emotion=None,
                smoothed_top_confidence=0.0,
                decision_source="SMOOTHED",
                geometry_emotion=None,
                geometry_strength=0.0,
                geometry_reason="no_signal",
                expression_intensity=0.0,
                final_emotion="uncertain",
                final_confidence=0.0,
                iot_emotion=self.iot_emotion,
                iot_hold_remaining=max(0.0, self.iot_hold_until - timestamp),
                iot_transition_decision="no_signal",
                triggered_rules=[],
                hold_active=timestamp < self.hold_until,
                hold_remaining=max(0.0, self.hold_until - timestamp),
                transition_decision="no_signal",
                uncertain=True,
                status="uncertain",
                debug_reason="empty_fer_scores",
                low_confidence_reason="empty_fer_scores",
                filtered_scores={},
            )

        raw_scores = normalize_scores(fer_scores)
        eyebrow_pos = float(landmarks.get("eyebrow_position", 0.0))

        # Angry correction: emphasize angry when brow geometry is strongly downward.
        if eyebrow_pos < -0.2:
            raw_scores["angry"] = float(raw_scores.get("angry", 0.0)) * 1.5
            raw_scores = normalize_scores(raw_scores)

        raw_top_emotion, raw_top_conf = pick_top(raw_scores)
        raw_second_emotion: Optional[str] = None
        raw_second_conf: float = 0.0
        ordered_raw = sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)
        if len(ordered_raw) > 1:
            raw_second_emotion = ordered_raw[1][0]
            raw_second_conf = float(ordered_raw[1][1])

        # Bias correction: slightly amplify strong non-neutral raw winners.
        if raw_top_emotion is not None and raw_top_emotion != "neutral":
            raw_top_conf = clamp01(raw_top_conf * 1.15)

        calibrated = self._calibrate_scores(raw_scores)
        quality_adjusted = self._apply_face_quality(calibrated, face_quality)

        low_conf = self._low_confidence_filter(quality_adjusted)
        filtered = low_conf.filtered_scores

        weighted = self._apply_weights(filtered)
        baseline = self._update_feature_baseline(landmarks) if raw_top_conf < 0.5 else None
        rule_adjusted, rules = apply_facial_rules(
            weighted,
            landmarks,
            baseline=baseline,
            raw_top_confidence=raw_top_conf,
        )
        smoothed = self._smooth(rule_adjusted)

        smoothed_top_emotion, smoothed_top_conf = pick_top(smoothed)
        geometry_pick, geometry_strength, geometry_reason = geometry_emotion(
            cast(Dict[str, float], landmarks),
            raw_top_confidence=raw_top_conf,
        )
        expression_intensity = float(landmarks.get("expression_intensity", 0.0))
        hold_break_reason = "none"
        decision_reason = "pipeline_default"

        if raw_top_emotion is not None:
            self.temporal_window.append(
                {
                    "emotion": raw_top_emotion,
                    "confidence": raw_top_conf,
                }
            )

        high_conf_override = bool(raw_top_emotion is not None and raw_top_conf >= HIGH_CONFIDENCE_OVERRIDE_THRESHOLD)
        if high_conf_override:
            stable_emotion = raw_top_emotion
            stable_scores = {stable_emotion: raw_top_conf} if stable_emotion else {}
            stable_confidence = raw_top_conf
            candidate_emotion = stable_emotion
            candidate_conf = stable_confidence
            final_emotion = stable_emotion or self.final_emotion_state
            final_conf = stable_confidence
            self.last_change_time = timestamp
            transition_decision = "high_conf_override"
            decision_source = "RAW_HIGH_CONF_OVERRIDE"
            decision_reason = "raw_top_confidence_override"
            rules.append("high_confidence_override")
            LOGGER.info(
                "High-confidence override applied: emotion=%s conf=%.3f threshold=%.2f",
                raw_top_emotion,
                raw_top_conf,
                HIGH_CONFIDENCE_OVERRIDE_THRESHOLD,
            )
        # STEP 6: ignore weak raw signal (unless geometry gives a strong signal).
        elif raw_top_conf < WEAK_SIGNAL_CONFIDENCE_THRESHOLD and geometry_pick is None:
            stable_emotion = self.last_emotion or self.final_emotion_state
            stable_scores = {stable_emotion: self.current_confidence} if stable_emotion else {}
            stable_confidence = clamp01(float(self.current_confidence))
            candidate_emotion = self.final_emotion_state
            candidate_conf = stable_confidence
            final_emotion = self.final_emotion_state
            final_conf = clamp01(float(self.current_confidence))
            transition_decision = "noise_floor_hold"
            decision_source = "NOISE_FLOOR"
            decision_reason = "raw_below_weak_signal_floor"
            rules.append("raw_below_floor_hold")
        else:
            stable_emotion, stable_scores = compute_stable_emotion(self.temporal_window)
            if stable_emotion is None:
                stable_emotion = raw_top_emotion
                stable_scores = {stable_emotion: raw_top_conf} if stable_emotion is not None else {}
            stable_confidence = clamp01(float(stable_scores.get(stable_emotion, raw_top_conf))) if stable_emotion else 0.0

            consensus_emotion, consensus_confidence, consensus_count = compute_temporal_consensus(
                self.temporal_window,
                strong_threshold=TEMPORAL_STRONG_EMOTION_THRESHOLD,
                min_frames=TEMPORAL_CONSENSUS_MIN_FRAMES,
            )
            if consensus_emotion is not None:
                stable_emotion = consensus_emotion
                stable_confidence = max(stable_confidence, consensus_confidence)
                stable_scores[stable_emotion] = stable_confidence
                transition_decision = "temporal_consensus_override"
                decision_reason = f"temporal_consensus:{consensus_emotion}:{consensus_count}"
                rules.append(f"temporal_consensus_override:{consensus_emotion}:{consensus_count}")

            # STEP 5: strong geometry signals override weak model predictions.
            if (
                geometry_pick is not None
                and raw_top_conf < GEOMETRY_APPLY_RAW_MAX_CONFIDENCE
                and geometry_strength > GEOMETRY_OVERRIDE_MIN_STRENGTH
            ):
                stable_emotion = geometry_pick
                stable_confidence = clamp01(max(stable_confidence, geometry_strength))
                stable_scores[stable_emotion] = stable_confidence
                rules.append(f"geometry_override:{geometry_reason}")
                decision_reason = f"geometry_override:{geometry_reason}"

            # STEP 4: anti-bias dampening when one emotion dominates too long.
            if self.last_final_emotion is not None and self.final_emotion_state == self.last_final_emotion:
                self.repeat_count += 1
            else:
                self.repeat_count = 0

            if self.repeat_count > int(TEMPORAL_WINDOW_FPS * 2):
                stable_confidence = clamp01(stable_confidence * 0.85)
                if stable_emotion is not None:
                    stable_scores[stable_emotion] = stable_confidence
                rules.append("dominance_dampened")

            if self.last_emotion is None:
                self.last_emotion = stable_emotion

            # STEP 2: controlled change detection.
            if stable_emotion != self.last_emotion:
                if stable_confidence > STABLE_CHANGE_CONFIDENCE_THRESHOLD:
                    candidate_emotion = stable_emotion
                    candidate_conf = stable_confidence
                    transition_decision = "change_allowed"
                else:
                    candidate_emotion = self.last_emotion
                    candidate_conf = float(stable_scores.get(candidate_emotion, self.current_confidence)) if candidate_emotion else 0.0
                    transition_decision = "change_blocked"
            else:
                candidate_emotion = self.last_emotion
                candidate_conf = float(stable_scores.get(candidate_emotion, stable_confidence)) if candidate_emotion else 0.0
                transition_decision = "stable_same"

            self.last_emotion = candidate_emotion

            # STEP 3: hold with strong override and explicit release behavior.
            final_emotion = self.final_emotion_state
            final_conf = stable_confidence
            decision_source = "HOLD"

            if final_emotion == "uncertain" and stable_emotion is not None:
                final_emotion = stable_emotion
                self.last_change_time = timestamp
                decision_source = "INIT"

            if stable_emotion is not None and final_emotion != stable_emotion:
                if stable_confidence > STABLE_CHANGE_CONFIDENCE_THRESHOLD and (timestamp - self.last_change_time) > FINAL_HOLD_SECONDS:
                    final_emotion = stable_emotion
                    self.last_change_time = timestamp
                    decision_source = "HOLD_SWITCH"
                    decision_reason = "hold_elapsed_and_stable_change"
                elif stable_confidence > (self.current_confidence + HOLD_BREAK_CONFIDENCE_DELTA) or stable_confidence > FORCE_BREAK_CONFIDENCE_THRESHOLD:
                    final_emotion = stable_emotion
                    self.last_change_time = timestamp
                    final_conf = stable_confidence
                    decision_source = "HOLD_BREAK"
                    hold_break_reason = (
                        "confidence_delta"
                        if stable_confidence > (self.current_confidence + HOLD_BREAK_CONFIDENCE_DELTA)
                        else "confidence_force_break"
                    )
                    transition_decision = f"hold_broken:{hold_break_reason}"
                    decision_reason = f"hold_broken:{hold_break_reason}"
                    LOGGER.info(
                        "Hold broken: prev_emotion=%s prev_conf=%.3f new_emotion=%s new_conf=%.3f reason=%s",
                        self.final_emotion_state,
                        self.current_confidence,
                        stable_emotion,
                        stable_confidence,
                        hold_break_reason,
                    )
                else:
                    decision_source = "HOLD_WAIT"
                    decision_reason = "hold_active_waiting"
            else:
                self.last_change_time = timestamp
                decision_source = "HOLD_REFRESH"
                decision_reason = "stable_refresh"

            final_conf = stable_confidence if final_emotion == stable_emotion else clamp01(final_conf)

        self.final_emotion_state = final_emotion
        self.last_final_emotion = final_emotion
        hold_elapsed = max(0.0, timestamp - self.last_change_time)
        hold_remaining = max(0.0, FINAL_HOLD_SECONDS - hold_elapsed)
        hold_active = hold_remaining > 0.0

        uncertain = low_conf.uncertain and final_conf < self.config.min_confidence_threshold
        debug_reason = low_conf.reason
        status = "ok"
        if uncertain:
            status = "uncertain"
            debug_reason = low_conf.reason

        if current_state and current_state.get("force_uncertain"):
            uncertain = True
            status = "uncertain"
            debug_reason = "forced_by_state"

        if movement_metrics:
            # Lightweight debug channel so caller can inspect scheduling context.
            if movement_metrics.get("time_trigger"):
                rules.append("time_trigger")
            if movement_metrics.get("movement_trigger"):
                rules.append("movement_trigger")
            if movement_metrics.get("landmark_trigger"):
                rules.append("landmark_trigger")

        if uncertain:
            # Recover from borderline uncertainty when the face is clear and
            # raw signal is still reasonably strong.
            if raw_top_emotion is not None and raw_top_conf >= 0.28 and face_quality >= 0.55:
                recovered_emotion = stable_emotion or raw_top_emotion
                final_emotion = recovered_emotion
                final_conf = max(float(final_conf), float(raw_top_conf) * 0.9)
                uncertain = False
                status = "ok"
                debug_reason = "uncertain_recovered_good_quality"
                decision_source = "UNCERTAIN_RECOVERED"
                decision_reason = "recovered_from_raw_with_good_face_quality"
                rules.append("uncertain_recovered_good_quality")
            else:
                final_emotion = "uncertain"
                decision_source = "UNCERTAIN"
                decision_reason = "uncertain_low_confidence"

        LOGGER.info(
            {
                "window_size": len(self.temporal_window),
                "scores": stable_scores,
                "raw": raw_scores,
                "raw_top": raw_top_emotion,
                "stable": stable_emotion,
                "candidate": candidate_emotion,
                "final": final_emotion,
                "hold_remaining": hold_remaining,
                "reason": decision_source,
                "decision_reason": decision_reason,
                "hold_break_reason": hold_break_reason,
                "override": high_conf_override,
            }
        )

        self.current_emotion = final_emotion
        self.current_confidence = final_conf

        iot_emotion, iot_hold_remaining, iot_transition_decision = self._update_iot_actuation(
            final_emotion=final_emotion,
            final_confidence=final_conf,
            uncertain=uncertain,
            now_ts=timestamp,
        )

        # Optional throttle: write every 2nd processed frame to limit file growth.
        self._log_frame_index += 1
        if self._log_frame_index % 2 == 0:
            log_emotion_frame(
                {
                    "raw": raw_top_emotion,
                    "stable": stable_emotion,
                    "final": final_emotion,
                    "confidence": final_conf,
                    "hold_time": hold_elapsed,
                    "repeat_count": self.repeat_count,
                    "ui_emotion": final_emotion,
                    "iot_emotion": iot_emotion,
                    "final_emotion": final_emotion,
                    "decision_source": decision_source,
                    "geometry_emotion": geometry_pick,
                    "geometry_strength": geometry_strength,
                    "raw_emotion": raw_top_emotion,
                    "raw_confidence": raw_top_conf,
                    "smoothed_emotion": smoothed_top_emotion,
                    "smoothed_confidence": smoothed_top_conf,
                    "expression_intensity": expression_intensity,
                    "rule_triggers": rules,
                    "face_quality": face_quality,
                    "decision_reason": decision_reason,
                    "hold_break_reason": hold_break_reason,
                    "high_conf_override": high_conf_override,
                }
            )

        LOGGER.debug("FER raw scores: %s", raw_scores)
        LOGGER.debug("Rule triggers: %s", rules)
        LOGGER.debug("Smoothed scores: %s", smoothed)
        LOGGER.debug("Final emotion: %s (%.3f) reason=%s", final_emotion, final_conf, debug_reason)

        return ProcessEmotionResult(
            raw_scores=raw_scores,
            calibrated_scores=calibrated,
            weighted_scores=weighted,
            rule_adjusted_scores=rule_adjusted,
            smoothed_scores=smoothed,
            raw_top_emotion=raw_top_emotion,
            raw_top_confidence=raw_top_conf,
            stable_emotion=stable_emotion,
            stable_confidence=stable_confidence,
            smoothed_top_emotion=smoothed_top_emotion,
            smoothed_top_confidence=smoothed_top_conf,
            decision_source=decision_source,
            geometry_emotion=geometry_pick,
            geometry_strength=geometry_strength,
            geometry_reason=geometry_reason,
            expression_intensity=expression_intensity,
            final_emotion=final_emotion,
            final_confidence=final_conf,
            iot_emotion=iot_emotion,
            iot_hold_remaining=iot_hold_remaining,
            iot_transition_decision=iot_transition_decision,
            triggered_rules=rules,
            hold_active=hold_active,
            hold_remaining=hold_remaining,
            transition_decision=transition_decision,
            uncertain=uncertain,
            status=status,
            debug_reason=debug_reason,
            low_confidence_reason=low_conf.reason,
            filtered_scores=filtered,
        )

    def evaluate(self, raw_scores: Dict[str, float], features: Dict[str, Any], now_ts: float, face_quality: float = 1.0) -> Dict[str, Any]:
        result = self.process_emotion_signal(
            fer_scores=raw_scores,
            landmarks=features,
            current_state=None,
            timestamp=now_ts,
            movement_metrics=None,
            face_quality=face_quality,
        )
        data = asdict(result)

        # Keep legacy keys expected by realtime loop and logger.
        data["raw_emotion"] = result.raw_top_emotion
        data["raw_confidence"] = result.raw_top_confidence
        data["smoothed_emotion"] = result.smoothed_top_emotion
        data["smoothed_confidence"] = result.smoothed_top_confidence
        data["rule_triggers"] = result.triggered_rules
        data["scores"] = result.raw_scores
        data["candidate_scores"] = self._blend_candidate_scores(
            raw_calibrated=result.calibrated_scores,
            corrected=result.rule_adjusted_scores,
            smoothed=result.smoothed_scores,
        )
        data["hold_remaining"] = result.hold_remaining
        data["stable_emotion"] = result.stable_emotion
        data["stable_confidence"] = round(clamp01(float(result.stable_confidence)), 2)
        data["final_confidence"] = round(float(result.final_confidence), 2)
        data["iot_emotion"] = result.iot_emotion
        data["iot_hold_remaining"] = result.iot_hold_remaining
        data["iot_transition_decision"] = result.iot_transition_decision
        data["face_quality"] = face_quality
        data["feature_baseline"] = self.feature_baseline
        data["quality_adjusted_scores"] = result.filtered_scores
        return data

    def _update_iot_actuation(
        self,
        final_emotion: str,
        final_confidence: float,
        uncertain: bool,
        now_ts: float,
    ) -> Tuple[str, float, str]:
        """Simple IoT stabilization layer to avoid actuator flicker."""
        if not self.config.iot_actuation_enabled:
            return final_emotion, 0.0, "disabled"

        if self.config.iot_block_uncertain and uncertain:
            return self.iot_emotion, max(0.0, self.config.iot_actuation_hold_seconds - (now_ts - self.iot_last_change)), "blocked_uncertain"

        if final_confidence < self.config.iot_min_confidence:
            return self.iot_emotion, max(0.0, self.config.iot_actuation_hold_seconds - (now_ts - self.iot_last_change)), "blocked_low_conf"

        if final_emotion == "uncertain":
            return self.iot_emotion, max(0.0, self.config.iot_actuation_hold_seconds - (now_ts - self.iot_last_change)), "blocked_uncertain_label"

        if self.iot_emotion == "neutral" and self.iot_last_change == 0.0:
            self.iot_emotion = final_emotion
            self.iot_confidence = final_confidence
            self.iot_last_change = now_ts
            return self.iot_emotion, self.config.iot_actuation_hold_seconds, "init"

        if final_emotion == self.iot_emotion:
            self.iot_confidence = final_confidence
            return self.iot_emotion, self.config.iot_actuation_hold_seconds, "stable_same"

        if (
            final_confidence > (self.iot_confidence + HOLD_BREAK_CONFIDENCE_DELTA)
            or final_confidence >= HIGH_CONFIDENCE_OVERRIDE_THRESHOLD
        ):
            self.iot_emotion = final_emotion
            self.iot_confidence = final_confidence
            self.iot_last_change = now_ts
            return self.iot_emotion, self.config.iot_actuation_hold_seconds, "hold_break_override"

        elapsed = now_ts - self.iot_last_change
        if elapsed > self.config.iot_actuation_hold_seconds:
            self.iot_emotion = final_emotion
            self.iot_confidence = final_confidence
            self.iot_last_change = now_ts
            return self.iot_emotion, self.config.iot_actuation_hold_seconds, "switch_after_hold"

        return self.iot_emotion, max(0.0, self.config.iot_actuation_hold_seconds - elapsed), "hold_active"

    def _select_candidate(
        self,
        smoothed_scores: Dict[str, float],
        corrected_scores: Dict[str, float],
        raw_scores: Dict[str, float],
    ) -> Tuple[Optional[str], float]:
        candidate_scores = self._blend_candidate_scores(
            raw_calibrated=raw_scores,
            corrected=corrected_scores,
            smoothed=smoothed_scores,
        )
        return pick_top(candidate_scores)

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

    def _low_confidence_filter(self, scores: Dict[str, float]) -> LowConfidenceResult:
        if not scores:
            return LowConfidenceResult(filtered_scores={}, status="uncertain", reason="empty_scores", uncertain=True)

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top1_name, top1_val = sorted_scores[0]
        top2_val = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
        gap = float(top1_val - top2_val)

        uncertain = False
        reasons: List[str] = []

        if top1_val < self.config.min_confidence_threshold:
            uncertain = True
            reasons.append("top_conf_below_threshold")

        if gap < self.config.confidence_gap_threshold:
            uncertain = True
            reasons.append("top1_top2_gap_too_small")

        filtered = dict(scores)
        if uncertain:
            # Keep explainable distribution but suppress overconfident winner.
            filtered[top1_name] = float(top1_val) * 0.92
            filtered = normalize_scores(filtered)
            return LowConfidenceResult(
                filtered_scores=filtered,
                status="uncertain",
                reason=";".join(reasons),
                uncertain=True,
            )

        return LowConfidenceResult(filtered_scores=filtered, status="accepted", reason="accepted", uncertain=False)

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

        if raw_conf < self.config.raw_override_confidence:
            return final_emotion, final_conf, False

        if self.raw_streak_count < self.config.raw_override_streak:
            return final_emotion, final_conf, False

        if (raw_conf - final_conf) < self.config.raw_override_margin:
            return final_emotion, final_conf, False

        self.current_emotion = raw_emotion
        self.current_confidence = raw_conf
        return raw_emotion, raw_conf, True

    def _calibrate_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        # Apply a power calibration curve then renormalize.
        calibrated: Dict[str, float] = {}
        for emotion in self.config.supported_emotions:
            value = clamp01(scores.get(emotion, 0.0))
            calibrated[emotion] = value ** self.config.confidence_calibration_power
        return normalize_scores(calibrated)

    def _apply_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        # Emotion priors live in config so behavior can be tuned without code edits.
        weighted: Dict[str, float] = {}
        for emotion, score in scores.items():
            weighted[emotion] = score * float(self.config.emotion_weights.get(emotion, 1.0))
        return normalize_scores(weighted)

    def _apply_face_quality(self, scores: Dict[str, float], quality: float) -> Dict[str, float]:
        # Poor-quality frames (blur, bad exposure) are often overconfident noise.
        adjusted = dict(scores)
        q = clamp01(quality)
        if q < 0.55:
            penalty = 1.0 - ((1.0 - q) * LOW_QUALITY_NON_NEUTRAL_PENALTY)
            for emotion in self.config.supported_emotions:
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

        weighted_average = normalize_scores(dict(aggregate))
        if not self.config.use_ema_smoothing:
            return weighted_average

        if self.ema_scores is None:
            self.ema_scores = dict(weighted_average)
            return weighted_average

        alpha = clamp01(self.config.ema_alpha)
        ema: Dict[str, float] = {}
        keys = set(self.ema_scores) | set(weighted_average)
        for emotion in keys:
            prev = float(self.ema_scores.get(emotion, 0.0))
            curr = float(weighted_average.get(emotion, 0.0))
            ema[emotion] = (alpha * curr) + ((1.0 - alpha) * prev)
        self.ema_scores = normalize_scores(ema)
        return dict(self.ema_scores)

    def _transition_control(
        self,
        candidate_emotion: Optional[str],
        candidate_conf: float,
        smoothed_scores: Dict[str, float],
        now_ts: float,
    ) -> Tuple[Optional[str], float, str]:
        if not candidate_emotion:
            return None, 0.0, "no_candidate"

        if self.current_emotion is None:
            self.candidate_streak_emotion = None
            self.candidate_streak_count = 0
            return candidate_emotion, candidate_conf, "initial_pick"

        # Keep current emotion unless the candidate exceeds the configured margin.
        current_score = float(smoothed_scores.get(self.current_emotion, self.current_confidence))
        if candidate_emotion == self.current_emotion:
            self.candidate_streak_emotion = None
            self.candidate_streak_count = 0
            return candidate_emotion, candidate_conf, "same_emotion"

        # Build short evidence streak before allowing frequent flips.
        if self.candidate_streak_emotion == candidate_emotion:
            self.candidate_streak_count += 1
        else:
            self.candidate_streak_emotion = candidate_emotion
            self.candidate_streak_count = 1

        # If candidate repeatedly wins, reduce transition friction.
        if self.candidate_streak_count >= 3 and candidate_conf >= (current_score + (self.transition_margin * 0.5)):
            self.last_transition_ts = now_ts
            return candidate_emotion, candidate_conf, "streak_confirmed"

        # Very strong candidates can preempt quickly.
        if candidate_conf >= 0.62 and candidate_conf > current_score:
            self.last_transition_ts = now_ts
            return candidate_emotion, candidate_conf, "strong_candidate"

        if (now_ts - self.last_transition_ts) < self.config.transition_cooldown_seconds:
            return self.current_emotion, current_score, "blocked_by_cooldown"

        if candidate_emotion != self.current_emotion and (candidate_conf - current_score) < self.transition_margin:
            return self.current_emotion, current_score, "blocked_by_margin"

        self.last_transition_ts = now_ts
        return candidate_emotion, candidate_conf, "switched"

    def _apply_hold(
        self,
        candidate_emotion: Optional[str],
        candidate_conf: float,
        now_ts: float,
        smoothed_scores: Dict[str, float],
    ) -> Tuple[str, float, bool, float]:
        if not candidate_emotion:
            if self.current_emotion:
                return self.current_emotion, self.current_confidence, now_ts < self.hold_until, max(0.0, self.hold_until - now_ts)
            return "uncertain", 0.0, False, 0.0

        if self.current_emotion is None:
            self.current_emotion = candidate_emotion
            self.current_confidence = candidate_conf
            self.hold_until = now_ts + self.hold_seconds
            return self.current_emotion, self.current_confidence, True, self.hold_seconds

        if candidate_emotion == self.current_emotion:
            # Refresh hold window while emotion is stable.
            self.current_confidence = candidate_conf
            self.hold_until = now_ts + self.hold_seconds
            return self.current_emotion, self.current_confidence, True, self.hold_seconds

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
            return self.current_emotion, self.current_confidence, True, self.hold_seconds

        if (
            now_ts < self.hold_until
            and candidate_emotion == "surprise"
            and candidate_conf >= HOLD_ESCAPE_STRONG_SURPRISE
            and candidate_conf > current_score
        ):
            self.current_emotion = candidate_emotion
            self.current_confidence = candidate_conf
            self.hold_until = now_ts + self.hold_seconds
            return self.current_emotion, self.current_confidence, True, self.hold_seconds

        if now_ts < self.hold_until and self.candidate_streak_count >= 3 and candidate_conf >= (current_score + self.config.hold_override_margin):
            self.current_emotion = candidate_emotion
            self.current_confidence = candidate_conf
            self.hold_until = now_ts + self.hold_seconds
            return self.current_emotion, self.current_confidence, True, self.hold_seconds

        if now_ts < self.hold_until and candidate_conf >= (current_score + self.config.hold_override_margin):
            self.current_emotion = candidate_emotion
            self.current_confidence = candidate_conf
            self.hold_until = now_ts + self.hold_seconds
            return self.current_emotion, self.current_confidence, True, self.hold_seconds

        if now_ts < self.hold_until and candidate_conf < (current_score + self.hold_switch_margin):
            return (
                self.current_emotion,
                self.current_confidence,
                True,
                max(0.0, self.hold_until - now_ts),
            )

        self.current_emotion = candidate_emotion
        self.current_confidence = candidate_conf
        self.hold_until = now_ts + self.hold_seconds
        return self.current_emotion, self.current_confidence, True, self.hold_seconds


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
        detector_short = solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.15)
        detector_full = solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.10)
        return {"mode": "solutions", "detectors": [detector_short, detector_full]}

    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = Path(__file__).resolve().parent / "models" / "face_detector.tflite"
        model_path = _ensure_model(model_path, _FACE_DETECTOR_MODEL_URL)

        options = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.10,
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
        if _FACE_DETECTOR_CONFIG.get("mode") == "none":
            LOGGER.error(
                "MediaPipe face detector is unavailable in this Python environment. "
                "Use the project venv interpreter to run realtime_emotion.py."
            )
        else:
            LOGGER.info("MediaPipe face detector initialized in mode=%s", _FACE_DETECTOR_CONFIG.get("mode"))
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


def _enhance_detection_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Improve contrast for detection without changing pipeline architecture."""
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.equalizeHist(y)
    enhanced = cv2.merge((y, cr, cb))
    return cv2.cvtColor(enhanced, cv2.COLOR_YCrCb2BGR)


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


def confidence_gap(scores: Dict[str, float]) -> float:
    if not scores:
        return 0.0
    ordered = sorted(scores.values(), reverse=True)
    if len(ordered) < 2:
        return float(ordered[0]) if ordered else 0.0
    return float(ordered[0] - ordered[1])


def geometry_emotion(features: Dict[str, float], raw_top_confidence: float = 0.0) -> Tuple[Optional[str], float, str]:
    """Infer a strong emotion directly from facial geometry cues.

    Returns (emotion, strength, reason). When no strong geometry pattern is
    present, returns (None, 0.0, "no_match").
    """
    if not features or not bool(features.get("available", False)):
        return None, 0.0, "no_landmarks"

    mouth_open_ratio = float(features.get("mouth_open_ratio", 0.0))
    eye_open_ratio = float(features.get("eye_open_ratio", 0.0))
    eyebrow_pos = float(features.get("eyebrow_position", 0.0))
    eyebrow_inner_up = float(features.get("eyebrow_inner_raise", 0.0))
    lip_spread_ratio_val = float(features.get("lip_spread_ratio", 0.0))

    mouth_open_norm = clamp01(mouth_open_ratio / 0.12)
    eyebrow_raise_norm = clamp01(max(0.0, eyebrow_inner_up) / 0.12)
    eyebrow_down_norm = clamp01(max(0.0, -eyebrow_pos) / 0.12)
    eyes_narrow_norm = clamp01(max(0.0, 0.12 - eye_open_ratio) / 0.12)
    lips_flat = lip_spread_ratio_val < SAD_LIP_SPREAD_MAX

    if mouth_open_norm > 0.6 and eyebrow_raise_norm > 0.5:
        return "surprise", 0.9, "rule_surprise_mouth_brow"

    if eyebrow_down_norm > 0.6 and eyes_narrow_norm > 0.5:
        return "angry", 0.85, "rule_angry_brow_eye"

    if eyebrow_raise_norm > 0.56 and eyes_narrow_norm > 0.45 and lips_flat:
        return "sad", 0.72, "rule_sad_brow_inner_lips_flat"

    if eye_open_ratio > 0.32 and mouth_open_ratio > 0.25:
        return "fear", 0.75, "fear_pattern"

    return None, 0.0, "no_match"


def compute_stable_emotion(window: Iterable[Dict[str, Any]]) -> Tuple[Optional[str], Dict[str, float]]:
    scores: Dict[str, float] = defaultdict(float)
    items = list(window)
    if not items:
        return None, {}

    window_length = max(1, len(items))
    total_weight = 0.0
    for index, item in enumerate(items):
        emotion = item.get("emotion")
        if not emotion:
            continue

        confidence = float(item.get("confidence", 0.0))
        recency_weight = 0.5 + ((index / window_length) * 0.5)
        weight = confidence * recency_weight
        scores[str(emotion)] += weight
        total_weight += weight

    if not scores:
        return None, {}

    for emotion in list(scores.keys()):
        scores[emotion] = float(scores[emotion] / (total_weight + 1e-6))

    stable_emotion = max(scores, key=lambda emotion: scores[emotion])
    return stable_emotion, dict(scores)


def compute_temporal_consensus(
    window: Iterable[Dict[str, Any]],
    strong_threshold: float = TEMPORAL_STRONG_EMOTION_THRESHOLD,
    min_frames: int = TEMPORAL_CONSENSUS_MIN_FRAMES,
) -> Tuple[Optional[str], float, int]:
    """Find repeated strong emotions in recent frames using weighted mode."""
    items = list(window)
    if not items:
        return None, 0.0, 0

    weighted_votes: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    confidences: Dict[str, float] = defaultdict(float)

    window_len = max(1, len(items))
    for index, item in enumerate(items):
        emotion = item.get("emotion")
        confidence = float(item.get("confidence", 0.0))
        if not emotion or confidence < strong_threshold:
            continue

        recency_weight = 0.65 + ((index / window_len) * 0.35)
        weighted_votes[str(emotion)] += confidence * recency_weight
        counts[str(emotion)] += 1
        confidences[str(emotion)] += confidence

    if not weighted_votes:
        return None, 0.0, 0

    consensus_emotion = max(weighted_votes, key=lambda emotion: weighted_votes[emotion])
    consensus_count = counts.get(consensus_emotion, 0)
    if consensus_count < min_frames:
        return None, 0.0, 0

    avg_confidence = confidences[consensus_emotion] / max(1, consensus_count)
    return consensus_emotion, clamp01(avg_confidence), consensus_count


def _point(landmarks: List[Any], index: int) -> Tuple[float, float]:
    item = landmarks[index]
    return float(item.x), float(item.y)


def _distance(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def _face_width(landmarks: List[Any]) -> float:
    return max(_distance(_point(landmarks, 234), _point(landmarks, 454)), 1e-6)


def _face_height(landmarks: List[Any]) -> float:
    ys = [float(point.y) for point in landmarks]
    if not ys:
        return 1e-6
    return max(max(ys) - min(ys), 1e-6)


def mouth_open_ratio(landmarks: List[Any]) -> float:
    mouth_open = _distance(_point(landmarks, 13), _point(landmarks, 14))
    return float(mouth_open / _face_height(landmarks))


def lip_spread_ratio(landmarks: List[Any]) -> float:
    mouth_width = max(_distance(_point(landmarks, 61), _point(landmarks, 291)), 1e-6)
    return float(mouth_width / _face_height(landmarks))


def eye_open_ratio(landmarks: List[Any]) -> float:
    left_eye_open = _distance(_point(landmarks, 159), _point(landmarks, 145))
    right_eye_open = _distance(_point(landmarks, 386), _point(landmarks, 374))
    return float(((left_eye_open + right_eye_open) * 0.5) / _face_height(landmarks))


def eyebrow_position(landmarks: List[Any]) -> float:
    left_eye_y = (_point(landmarks, 159)[1] + _point(landmarks, 145)[1]) * 0.5
    right_eye_y = (_point(landmarks, 386)[1] + _point(landmarks, 374)[1]) * 0.5
    left_brow_y = (_point(landmarks, 70)[1] + _point(landmarks, 63)[1] + _point(landmarks, 105)[1]) / 3.0
    right_brow_y = (_point(landmarks, 300)[1] + _point(landmarks, 293)[1] + _point(landmarks, 334)[1]) / 3.0
    left_delta = left_eye_y - left_brow_y
    right_delta = right_eye_y - right_brow_y
    return float(((left_delta + right_delta) * 0.5) / _face_height(landmarks))


def eyebrow_inner_raise(landmarks: List[Any]) -> float:
    left_eye_y = (_point(landmarks, 159)[1] + _point(landmarks, 145)[1]) * 0.5
    right_eye_y = (_point(landmarks, 386)[1] + _point(landmarks, 374)[1]) * 0.5
    left_inner_y = (_point(landmarks, 107)[1] + _point(landmarks, 66)[1]) * 0.5
    right_inner_y = (_point(landmarks, 336)[1] + _point(landmarks, 296)[1]) * 0.5
    left_delta = left_eye_y - left_inner_y
    right_delta = right_eye_y - right_inner_y
    return float(((left_delta + right_delta) * 0.5) / _face_height(landmarks))


def smile_ratio(landmarks: List[Any]) -> float:
    mouth_center_y = (_point(landmarks, 13)[1] + _point(landmarks, 14)[1]) * 0.5
    corner_y = (_point(landmarks, 61)[1] + _point(landmarks, 291)[1]) * 0.5
    return float((mouth_center_y - corner_y) / _face_height(landmarks))


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
        ox = float(relative_box.origin_x)
        oy = float(relative_box.origin_y)
        bw = float(relative_box.width)
        bh = float(relative_box.height)

        # Some builds may expose normalized values here; handle both forms.
        if bw <= 2.0 and bh <= 2.0 and ox <= 1.0 and oy <= 1.0:
            x1 = int(ox * frame_w)
            y1 = int(oy * frame_h)
            x2 = x1 + int(bw * frame_w)
            y2 = y1 + int(bh * frame_h)
        else:
            x1 = int(ox)
            y1 = int(oy)
            x2 = x1 + int(bw)
            y2 = y1 + int(bh)
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
    mode = detector_config.get("mode")
    if mode == "none":
        return []

    frame_h, frame_w = frame.shape[:2]

    def _run_detectors(frame_bgr: np.ndarray) -> List[Any]:
        rgb = _frame_to_rgb(frame_bgr)
        if mode == "solutions":
            dets: List[Any] = []
            detector_list = cast(List[Any], detector_config.get("detectors", []))
            for det in detector_list:
                result = det.process(rgb)
                if result and result.detections:
                    dets.extend(result.detections)
            return dets

        detector = detector_config.get("detector")
        if detector is None:
            return []
        mp_module = cast(Any, mp)
        mp_image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        return result.detections if result and result.detections else []

    def _detections_to_boxes(
        detections: List[Any],
        det_w: int,
        det_h: int,
        scale_back_x: float = 1.0,
        scale_back_y: float = 1.0,
    ) -> List[Tuple[int, int, int, int, float]]:
        boxes_local: List[Tuple[int, int, int, int, float]] = []
        for det in detections:
            location_data = getattr(det, "location_data", None)
            relative_box = getattr(location_data, "relative_bounding_box", None) if location_data else None
            if relative_box is None:
                relative_box = getattr(det, "bounding_box", None)

            box = _relative_bbox_to_xyxy(relative_box, det_w, det_h)
            if box is None:
                continue

            x1, y1, x2, y2 = box
            if scale_back_x != 1.0 or scale_back_y != 1.0:
                x1 = int(x1 / scale_back_x)
                y1 = int(y1 / scale_back_y)
                x2 = int(x2 / scale_back_x)
                y2 = int(y2 / scale_back_y)
                scaled = clamp_box(x1, y1, x2, y2, frame_w, frame_h)
                if scaled is None:
                    continue
                x1, y1, x2, y2 = scaled

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

            boxes_local.append((x1, y1, x2, y2, score))
        return boxes_local

    # Pass 1: direct frame.
    detections = _run_detectors(frame)
    boxes = _detections_to_boxes(detections, frame_w, frame_h)

    # Pass 2: contrast-enhanced frame.
    if not boxes:
        enhanced = _enhance_detection_frame(frame)
        detections = _run_detectors(enhanced)
        boxes = _detections_to_boxes(detections, frame_w, frame_h)

    # Pass 3: upscaled enhanced frame for small/far faces.
    if not boxes:
        upscaled = cv2.resize(_enhance_detection_frame(frame), (frame_w * 2, frame_h * 2), interpolation=cv2.INTER_LINEAR)
        detections = _run_detectors(upscaled)
        boxes = _detections_to_boxes(detections, frame_w * 2, frame_h * 2, scale_back_x=2.0, scale_back_y=2.0)

    boxes.sort(key=lambda item: item[4], reverse=True)
    return boxes[:MAX_FACES_PROCESS]


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


def get_face_landmarks_mediapipe(frame: np.ndarray, mediapipe_config: Optional[Dict[str, Any]] = None) -> Optional[List[Any]]:
    """Public wrapper for obtaining first-face landmarks from MediaPipe."""
    return _face_landmarks(frame, mediapipe_config)


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
        mouth_open = mouth_open_ratio(landmarks)
        lip_spread = lip_spread_ratio(landmarks)
        eyes_open = eye_open_ratio(landmarks)
        brows_pos = eyebrow_position(landmarks)
        brows_inner_raise = eyebrow_inner_raise(landmarks)
        smile = smile_ratio(landmarks)

        expression_intensity = (
            abs(mouth_open - float(MOUTH_OPEN_THRESHOLD))
            + abs(eyes_open - float(EYE_OPEN_THRESHOLD))
            + abs(brows_inner_raise - float(EYEBROW_INNER_RAISE_THRESHOLD))
            + abs(smile - float(SMILE_RATIO_THRESHOLD))
        ) * 0.25

        features.update(
            {
                "mouth_open_ratio": float(mouth_open),
                "lip_spread_ratio": float(lip_spread),
                "eye_open_ratio": float(eyes_open),
                "eyebrow_position": float(brows_pos),
                "eyebrow_inner_raise": float(brows_inner_raise),
                "smile_ratio": float(smile),
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


def landmark_change_delta(
    previous: Optional[Tuple[float, float, float, float]],
    current: Optional[Tuple[float, float, float, float]],
) -> Tuple[float, float]:
    if previous is None or current is None:
        return 0.0, 0.0
    deltas = [abs(a - b) for a, b in zip(previous, current)]
    return float(sum(deltas)), float(max(deltas) if deltas else 0.0)


def face_box_movement_delta(
    previous_box: Optional[Tuple[int, int, int, int]],
    current_box: Optional[Tuple[int, int, int, int]],
) -> float:
    if previous_box is None or current_box is None:
        return 0.0

    prev_cx = (previous_box[0] + previous_box[2]) * 0.5
    prev_cy = (previous_box[1] + previous_box[3]) * 0.5
    curr_cx = (current_box[0] + current_box[2]) * 0.5
    curr_cy = (current_box[1] + current_box[3]) * 0.5
    return float(math.hypot(curr_cx - prev_cx, curr_cy - prev_cy))


def landmark_changed(
    previous: Optional[Tuple[float, float, float, float]],
    current: Optional[Tuple[float, float, float, float]],
    threshold: float = LANDMARK_CHANGE_THRESHOLD,
) -> bool:
    # L1 distance is enough here and cheaper than more complex metrics.
    delta, max_delta = landmark_change_delta(previous, current)
    return delta >= threshold and max_delta >= (threshold * 0.35)


def face_movement_detected(
    previous_box: Optional[Tuple[int, int, int, int]],
    current_box: Optional[Tuple[int, int, int, int]],
    movement_threshold_px: float,
) -> bool:
    # Compare center-point motion between previous and current face boxes.
    return face_box_movement_delta(previous_box, current_box) >= movement_threshold_px


def apply_facial_rules(
    scores: Dict[str, float],
    features: Dict[str, Any],
    baseline: Optional[Dict[str, float]] = None,
    raw_top_confidence: float = 0.0,
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
    strongest_non_neutral = max(
        (float(score) for emotion, score in adjusted.items() if emotion != "neutral"),
        default=0.0,
    )
    if (
        expression_intensity <= LOW_EXPRESSION_INTENSITY_MAX
        and neutral_support >= 0.35
        and strongest_non_neutral < 0.45
        and raw_top_confidence < 0.4
    ):
        adjusted["neutral"] = adjusted.get("neutral", 0.0) + NEUTRAL_GUARD_BOOST
        adjusted["sad"] = adjusted.get("sad", 0.0) * 0.86
        adjusted["angry"] = adjusted.get("angry", 0.0) * 0.90
        triggers.append("neutral_guard_low_expression")
        neutral_guard_active = True

    surprise_prior = adjusted.get("surprise", 0.0)
    fear_prior = adjusted.get("fear", 0.0)
    angry_prior = adjusted.get("angry", 0.0)
    sad_prior = adjusted.get("sad", 0.0)

    strong_surprise_geometry = (
        mouth_val >= SURPRISE_MOUTH_BOOST_THRESHOLD
        and inner_brow_val >= SURPRISE_BROW_RAISE_THRESHOLD
        and eyes_open
    )

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
        if strong_surprise_geometry:
            surprise_boost += 0.18
        adjusted["surprise"] = max(adjusted.get("surprise", 0.0) + surprise_boost, fear_prior * 0.92)
        adjusted["fear"] = adjusted.get("fear", 0.0) * 0.93
        adjusted["angry"] = adjusted.get("angry", 0.0) * 0.92
        adjusted["sad"] = adjusted.get("sad", 0.0) * 0.90
        triggers.append("surprise_boost_mouth_open_brow_raised")

    # Secondary surprise cue: mouth open + eyes open, but only when surprise/fear is plausible.
    if mouth_open and eyes_open and max(surprise_prior, fear_prior) >= RULE_MIN_SURPRISE_OR_FEAR_SCORE:
        adjusted["surprise"] = adjusted.get("surprise", 0.0) + 0.18
        triggers.append("surprise_boost_mouth_open_eyes_open")

    # Angry heuristic: lowered brows + narrowed eyes with stronger angry support than sad.
    strong_angry_geometry = eyebrows_down and eyes_narrow and (not inner_raised) and mouth_val <= ANGRY_MOUTH_OPEN_MAX
    if eyebrows_down and eyes_narrow:
        adjusted["angry"] = adjusted.get("angry", 0.0) + 0.25
        triggers.append("angry_boost_simple_geometry")

    if (
        strong_angry_geometry
        and adjusted.get("angry", 0.0) >= (RULE_MIN_ANGRY_SCORE * 0.8)
        and angry_prior >= (sad_prior - 0.02)
    ):
        adjusted["angry"] = adjusted.get("angry", 0.0) + 0.28
        adjusted["sad"] = adjusted.get("sad", 0.0) * 0.84
        triggers.append("angry_boost_eyebrows_down_eyes_narrow")

    # Sad heuristic: stricter gating to avoid over-triggering on ambiguous geometry.
    sad_face_shape = (smile_val <= SAD_SMILE_MAX) and (lip_spread_val <= SAD_LIP_SPREAD_MAX)
    strong_sad_geometry = inner_raised and eyes_narrow and (not eyebrows_down) and sad_face_shape
    sad_blocked_by_competitors = surprise_prior >= (sad_prior + 0.03) or angry_prior >= (sad_prior + 0.03)
    if inner_raised and eyes_narrow and sad_face_shape and not sad_blocked_by_competitors:
        adjusted["sad"] = adjusted.get("sad", 0.0) + 0.12
        triggers.append("sad_boost_brows_up_lips_flat_strict")

    if (
        (not neutral_guard_active)
        and strong_sad_geometry
        and adjusted.get("sad", 0.0) >= RULE_MIN_SAD_SCORE
        and sad_prior >= (angry_prior + 0.05)
        and sad_prior >= (surprise_prior + 0.04)
        and not sad_blocked_by_competitors
    ):
        adjusted["sad"] = adjusted.get("sad", 0.0) + 0.14
        adjusted["angry"] = adjusted.get("angry", 0.0) * 0.90
        adjusted["surprise"] = adjusted.get("surprise", 0.0) * 0.94
        triggers.append("sad_boost_inner_brows_raised_low_eyes_strict")

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
    "EmotionEngineConfig",
    "EmotionHistory",
    "EmotionIntelligenceEngine",
    "LowConfidenceResult",
    "ProcessEmotionResult",
    "analyze_emotion",
    "analyze_emotion_fer",
    "apply_emotion_rules",
    "cache_last_result",
    "face_box_movement_delta",
    "confidence_gap",
    "crop_face",
    "compute_face_quality",
    "compute_stable_emotion",
    "decide_emotion_from_scores",
    "detect_emotion_fer",
    "detect_faces_mediapipe",
    "extract_face",
    "extract_features_mediapipe",
    "face_movement_detected",
    "fuse_emotion_scores",
    "get_face_detector_config",
    "get_face_landmarks_mediapipe",
    "get_face_mesh_config",
    "get_final_emotion",
    "geometry_emotion",
    "landmark_change_delta",
    "landmark_changed",
    "landmark_signature",
    "mouth_open_ratio",
    "eye_open_ratio",
    "eyebrow_position",
    "smile_ratio",
    "log_rate_limited_warning",
    "preprocess_face",
    "print_score_comparison",
    "should_run_detection",
    "smooth_emotions",
]
