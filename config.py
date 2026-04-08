"""Central configuration for the real-time emotion detection system."""

from __future__ import annotations

import cv2

# Camera and display
CAMERA_INDEX = 0
DISPLAY_FRAME_WIDTH = 960
SHOW_FPS = True

# Frame handling and scheduling
DETECTION_FRAME_SIZE = (320, 240)  # (width, height)
HEAVY_DETECTION_INTERVAL_SECONDS = 1.0
FACE_MOVEMENT_THRESHOLD_PX = 18.0
LANDMARK_CHANGE_THRESHOLD = 0.05

# Face detection and crop
MIN_FACE_SIZE = 20
MAX_FACES_PROCESS = 1
FACE_PADDING = 18
FACE_CROP_SIZE = (128, 128)

# Emotion model and temporal intelligence
SUPPORTED_EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
SMOOTHING_WINDOW_SIZE = 8
MIN_CONFIDENCE_THRESHOLD = 0.12
LOW_CONFIDENCE_GAP_THRESHOLD = 0.08
TRANSITION_MARGIN = 0.09
TRANSITION_COOLDOWN_SECONDS = 0.35
EMOTION_HOLD_SECONDS = 2.0
HOLD_SWITCH_MARGIN = 0.20
HOLD_OVERRIDE_MARGIN = 0.26
FEATURE_BASELINE_ALPHA = 0.92
USE_EMA_SMOOTHING = True
EMA_ALPHA = 0.40

# IoT actuation stabilization (separate from on-screen label stability).
IOT_ACTUATION_ENABLED = True
IOT_ACTUATION_HOLD_SECONDS = 3.0
IOT_MIN_CONFIDENCE = 0.42
IOT_CONFIRM_STREAK = 2
IOT_OVERRIDE_MARGIN = 0.18
IOT_BLOCK_UNCERTAIN = True

# Box stabilization controls (reduce rectangle flicker/jitter).
BOX_SMOOTHING_ALPHA = 0.78
BOX_HOLD_MISSING_SECONDS = 0.22

# Rare-emotion weighting and confidence calibration
EMOTION_WEIGHTS = {
    "happy": 1.02,
    "sad": 1.08,
    "angry": 1.20,
    "surprise": 1.10,
    "fear": 0.92,
    "disgust": 1.40,
    "neutral": 0.90,
}
CONFIDENCE_CALIBRATION_POWER = 0.92
LOW_QUALITY_NON_NEUTRAL_PENALTY = 0.65
LOW_QUALITY_NEUTRAL_BOOST = 0.28

# MediaPipe FaceMesh thresholds used by rule engine
MOUTH_OPEN_THRESHOLD = 0.20
EYE_OPEN_THRESHOLD = 0.080
EYE_NARROW_THRESHOLD = 0.050
EYEBROW_DOWN_THRESHOLD = 0.070
EYEBROW_INNER_RAISE_THRESHOLD = 0.090
SMILE_RATIO_THRESHOLD = 0.060
LIP_SPREAD_RATIO_THRESHOLD = 0.34

# Landmark expression gates
SURPRISE_MOUTH_BOOST_THRESHOLD = 0.32
SURPRISE_BROW_RAISE_THRESHOLD = 0.108
ANGRY_MOUTH_OPEN_MAX = 0.22
LOW_EXPRESSION_INTENSITY_MAX = 0.20
SAD_SMILE_MAX = 0.025
SAD_LIP_SPREAD_MAX = 0.32

# Rule confidence guards to avoid over-correction when geometry is noisy.
RULE_MIN_ANGRY_SCORE = 0.10
RULE_MIN_SAD_SCORE = 0.10
RULE_MIN_SURPRISE_OR_FEAR_SCORE = 0.09
RULE_MIN_HAPPY_SCORE = 0.30
NEUTRAL_GUARD_BOOST = 0.18
HOLD_ESCAPE_STRONG_HAPPY = 0.52
HOLD_ESCAPE_STRONG_SURPRISE = 0.56

# When raw FER is highly confident and consistent, allow it to override
# post-processing to avoid over-smoothing drift.
RAW_OVERRIDE_CONFIDENCE = 0.60
RAW_OVERRIDE_MARGIN = 0.10
RAW_OVERRIDE_STREAK = 2

# UI style (BGR colors)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.58
TEXT_THICKNESS = 2
TEXT_COLOR = (235, 235, 235)
BG_COLOR = (32, 32, 32)
BG_PADDING = 6
BOX_THICKNESS = 2
DEFAULT_EMOTION_COLOR = (160, 160, 160)

# Required color mapping from the task
EMOTION_COLORS = {
    "happy": (0, 220, 0),
    "sad": (255, 0, 0),
    "angry": (0, 0, 255),
    "surprise": (0, 255, 255),
    "neutral": (160, 160, 160),
    "fear": (0, 190, 255),
    "disgust": (80, 180, 90),
    "uncertain": (180, 180, 180),
}

# Ensemble weights (new)
ENSEMBLE_WEIGHTS = {
    'fer': 0.6,
    'deepface': 0.4
}

# Logging
DEBUG_MODE = False
LOG_LEVEL = "INFO"
ENABLE_JSONL_LOGGING = True
JSONL_LOG_PATH = "logs/emotion_events.jsonl"

# Benchmark mode
DEFAULT_BENCHMARK_SECONDS = 30
BENCHMARK_REPORT_PATH = "logs/benchmark_report.json"
