"""
PHASE 3: MEDIAPIPE FEATURE EXTRACTION

This module extracts facial features using MediaPipe FaceMesh:
- Mouth openness (smile detection)
- Eye openness (alertness)
- Eyebrow position (arousal)
- Head position / gaze direction

Features are used for rule-based corrections (PHASE 4).
"""

import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import cv2
import numpy as np

# Suppress TensorFlow startup noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from config import (
    EYE_OPENNESS_THRESHOLD,
    EYEBROW_RAISE_THRESHOLD,
    MOUTH_OPENNESS_THRESHOLD,
)


# ============================================================================
# MEDIAPIPE MODEL SETUP
# ============================================================================

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


class TasksFaceMeshProcessor:
    """Wrapper for MediaPipe Tasks FaceLandmarker."""

    def __init__(self, mp_module, landmarker):
        """
        Initialize the Tasks-based processor.
        
        Args:
            mp_module: MediaPipe module
            landmarker: FaceLandmarker vision task object
        """
        self._mp = mp_module
        self._landmarker = landmarker
        self._timestamp_ms = 0

    def process(self, frame_bgr: np.ndarray):
        """
        Process a frame to extract face landmarks.
        
        Args:
            frame_bgr: Input image in BGR format
        
        Returns:
            FaceLandmarkerResult with detected faces
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        self._timestamp_ms += 33
        return self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

    def close(self):
        """Close the landmarker and free resources."""
        self._landmarker.close()


def _ensure_face_landmarker_model(model_path: Path) -> Path:
    """
    Download the FaceLandmarker model if missing locally.
    
    Args:
        model_path: Path where model should be stored
    
    Returns:
        Path to the model file
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        return model_path

    print("Downloading MediaPipe FaceLandmarker model...")
    urllib.request.urlretrieve(MODEL_URL, str(model_path))
    print(f"Model downloaded to: {model_path}")
    return model_path


def _build_tasks_backend(mp):
    """
    Create a MediaPipe Tasks face-landmarker backend.
    
    Args:
        mp: MediaPipe module
    
    Returns:
        Dictionary with mode and processor
    """
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        env_model = os.environ.get("MP_FACE_LANDMARKER_MODEL")
        default_model = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
        model_path = Path(env_model) if env_model else default_model
        model_path = _ensure_face_landmarker_model(model_path)

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=5,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
        processor = TasksFaceMeshProcessor(mp, landmarker)
        return {
            "mode": "tasks",
            "processor": processor,
        }
    except Exception as e:
        return None


def init_mediapipe_facemesh():
    """
    Initialize FaceMesh backend using Solutions API (preferred) or Tasks fallback.
    
    Returns:
        Dictionary with initialized MediaPipe components or "none" mode if unavailable
    """
    try:
        import mediapipe as mp

        # Try classic solutions API (preferred)
        solutions = getattr(mp, "solutions", None)
        if solutions is None:
            try:
                from mediapipe import solutions as mp_solutions
                solutions = mp_solutions
            except Exception:
                solutions = None

        if solutions and hasattr(solutions, "face_mesh"):
            return {
                "mode": "solutions",
                "face_mesh": solutions.face_mesh.FaceMesh(),
                "drawing": solutions.drawing_utils,
            }

        # Fallback to Tasks API
        tasks_backend = _build_tasks_backend(mp)
        if tasks_backend:
            return tasks_backend

        return {"mode": "none"}

    except Exception as exc:
        print(f"MediaPipe initialization failed: {exc}")
        return {"mode": "none"}


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features_mediapipe(frame: np.ndarray,
                               mediapipe_config: Dict) -> Dict[str, any]:
    """
    Extract facial features from a frame using MediaPipe.
    
    PHASE 3 implementation. Features extracted:
    - mouth_openness: 0-1 (0=closed, 1=wide open)
    - eye_openness: 0-1 (0=closed, 1=wide open)
    - eyebrow_raise: 0-1 (0=relaxed, 1=raised high)
    - head_tilt: -1 to 1 (negative=left, positive=right)
    - gaze_direction: "left", "center", or "right"
    
    Args:
        frame: Input image (BGR, numpy array)
        mediapipe_config: Configuration dict from init_mediapipe_facemesh()
    
    Returns:
        Dictionary of feature names to values
    
    Example:
        >>> config = init_mediapipe_facemesh()
        >>> features = extract_features_mediapipe(frame, config)
        >>> if features["mouth_openness"] > 0.3:
        ...     print("Mouth is open")
    """
    features = {
        "mouth_openness": 0.0,
        "eye_openness": 0.0,
        "eyebrow_raise": 0.0,
        "head_tilt": 0.0,
        "gaze_direction": "center",
    }
    
    if mediapipe_config.get("mode") == "none":
        return features
    
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame.shape[:2]
        
        # Process based on backend
        if mediapipe_config.get("mode") == "solutions":
            face_mesh = mediapipe_config["face_mesh"]
            results = face_mesh.process(frame_rgb)
            if not results.multi_face_landmarks:
                return features
            landmarks = results.multi_face_landmarks[0].landmark
        
        elif mediapipe_config.get("mode") == "tasks":
            processor = mediapipe_config["processor"]
            results = processor.process(frame)
            if not results or not results.face_landmarks:
                return features
            landmarks = results.face_landmarks[0]
        
        else:
            return features
        
        # Extract individual features
        features["mouth_openness"] = _compute_mouth_openness(landmarks)
        features["eye_openness"] = _compute_eye_openness(landmarks)
        features["eyebrow_raise"] = _compute_eyebrow_raise(landmarks)
        features["head_tilt"] = _compute_head_tilt(landmarks, frame_w)
        features["gaze_direction"] = _compute_gaze_direction(landmarks)
    
    except Exception as e:
        # Silently handle errors
        pass
    
    return features


def _compute_mouth_openness(landmarks) -> float:
    """
    Compute mouth openness as ratio of vertical lip distance.
    
    Args:
        landmarks: MediaPipe face landmarks
    
    Returns:
        Mouth openness (0-1)
    """
    try:
        # Key points: 13=upper lip, 14=lower lip (center)
        upper_lip_y = landmarks[13].y
        lower_lip_y = landmarks[14].y
        
        mouth_open = abs(upper_lip_y - lower_lip_y)
        
        # Empirical normalization
        normalized = min(1.0, mouth_open / 0.05)
        return normalized
    except:
        return 0.0


def _compute_eye_openness(landmarks) -> float:
    """
    Compute eye openness as average of both eyes.
    
    Args:
        landmarks: MediaPipe face landmarks
    
    Returns:
        Eye openness (0-1)
    """
    try:
        # Left eye: 159=top, 145=bottom
        left_top = landmarks[159].y
        left_bottom = landmarks[145].y
        left_open = abs(left_top - left_bottom)
        
        # Right eye: 386=top, 374=bottom
        right_top = landmarks[386].y
        right_bottom = landmarks[374].y
        right_open = abs(right_top - right_bottom)
        
        avg_open = (left_open + right_open) / 2
        
        # Empirical normalization
        normalized = min(1.0, avg_open / 0.03)
        return normalized
    except:
        return 0.0


def _compute_eyebrow_raise(landmarks) -> float:
    """
    Compute eyebrow raise as vertical displacement.
    
    Args:
        landmarks: MediaPipe face landmarks
    
    Returns:
        Eyebrow raise (0-1)
    """
    try:
        # Left eyebrow: 107=inner, 66=outer
        left_inner = landmarks[107].y
        left_outer = landmarks[66].y
        
        # Right eyebrow: 336=inner, 296=outer
        right_inner = landmarks[336].y
        right_outer = landmarks[296].y
        
        # Raised eyebrows have smaller y values (closer to top)
        raise_amount = abs((left_inner + right_inner) / 2 - 
                          (left_outer + right_outer) / 2)
        
        # Empirical normalization
        normalized = min(1.0, raise_amount / 0.03)
        return normalized
    except:
        return 0.0


def _compute_head_tilt(landmarks, frame_w: int) -> float:
    """
    Compute head tilt (left-right rotation).
    
    Args:
        landmarks: MediaPipe face landmarks
        frame_w: Frame width
    
    Returns:
        Head tilt (-1 to 1, where 0 is straight)
    """
    try:
        # Landmark 4 = nose tip
        nose_x = landmarks[4].x
        
        # Normalize to -1 (left) to 1 (right)
        tilt = (nose_x - 0.5) * 2
        return tilt
    except:
        return 0.0


def _compute_gaze_direction(landmarks) -> str:
    """
    Estimate where person is looking.
    
    Args:
        landmarks: MediaPipe face landmarks
    
    Returns:
        One of: "left", "center", "right"
    """
    try:
        # Landmarks 159 & 386 = eye positions
        left_eye_x = landmarks[159].x
        right_eye_x = landmarks[386].x
        avg_eye_x = (left_eye_x + right_eye_x) / 2
        
        if avg_eye_x < 0.4:
            return "left"
        elif avg_eye_x > 0.6:
            return "right"
        else:
            return "center"
    except:
        return "center"


# ============================================================================
# FEATURE ANALYSIS UTILITIES
# ============================================================================

def has_open_mouth(features: Dict[str, float],
                   threshold: float = MOUTH_OPENNESS_THRESHOLD) -> bool:
    """
    Check if mouth is open.
    
    Args:
        features: Feature dictionary from extract_features_mediapipe()
        threshold: Openness threshold
    
    Returns:
        True if mouth is open
    """
    return features.get("mouth_openness", 0.0) > threshold


def has_wide_eyes(features: Dict[str, float],
                  threshold: float = EYE_OPENNESS_THRESHOLD) -> bool:
    """
    Check if eyes are wide open (surprise, fear).
    
    Args:
        features: Feature dictionary
        threshold: Openness threshold
    
    Returns:
        True if eyes are wide
    """
    return features.get("eye_openness", 0.0) > threshold


def has_raised_eyebrows(features: Dict[str, float],
                        threshold: float = EYEBROW_RAISE_THRESHOLD) -> bool:
    """
    Check if eyebrows are raised.
    
    Args:
        features: Feature dictionary
        threshold: Raise threshold
    
    Returns:
        True if eyebrows are raised
    """
    return features.get("eyebrow_raise", 0.0) > threshold


def is_looking_away(features: Dict[str, float]) -> bool:
    """
    Check if person is looking away (not at camera).
    
    Args:
        features: Feature dictionary
    
    Returns:
        True if not looking at camera
    """
    gaze = features.get("gaze_direction", "center")
    return gaze in ["left", "right"]
