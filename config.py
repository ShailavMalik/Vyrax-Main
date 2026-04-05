"""Central configuration for the real-time emotion detection system."""

from __future__ import annotations

import cv2

# Camera and display
CAMERA_INDEX = 0
DISPLAY_FRAME_WIDTH = 960
SHOW_FPS = True

# Frame handling and scheduling
DETECTION_FRAME_SIZE = (320, 240)  # (width, height)
HEAVY_DETECTION_INTERVAL_SECONDS = 2.0
FACE_MOVEMENT_THRESHOLD_PX = 18.0
LANDMARK_CHANGE_THRESHOLD = 0.07

# Face detection and crop
MIN_FACE_SIZE = 50
MAX_FACES_PROCESS = 1
FACE_PADDING = 18
FACE_CROP_SIZE = (128, 128)

# Emotion model and temporal intelligence
SUPPORTED_EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
SMOOTHING_WINDOW_SIZE = 8
MIN_CONFIDENCE_THRESHOLD = 0.12
TRANSITION_MARGIN = 0.09
EMOTION_HOLD_SECONDS = 2.0
HOLD_SWITCH_MARGIN = 0.20
FEATURE_BASELINE_ALPHA = 0.92

# Box stabilization controls (reduce rectangle flicker/jitter).
BOX_SMOOTHING_ALPHA = 0.78
BOX_HOLD_MISSING_SECONDS = 0.22

# Rare-emotion weighting and confidence calibration
EMOTION_WEIGHTS = {
    "happy": 1.06,
    "sad": 1.08,
    "angry": 1.12,
    "surprise": 1.08,
    "fear": 1.08,
    "disgust": 1.20,
    "neutral": 0.96,
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

# Logging
DEBUG_MODE = False
LOG_LEVEL = "INFO"
ENABLE_JSONL_LOGGING = True
JSONL_LOG_PATH = "logs/emotion_events.jsonl"

# Benchmark mode
DEFAULT_BENCHMARK_SECONDS = 30
BENCHMARK_REPORT_PATH = "logs/benchmark_report.json"