"""Compatibility wrapper for the emotion engine.

The real implementation now lives in emotion_engine.py. This file remains so
older imports keep working without changes.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from emotion_engine import *  # noqa: F401,F403

LOGGER = logging.getLogger("detection")

VIT_MODEL_NAME = "carlosleao/RAFDB-Facial-Expression-Recognition"
STANDARD_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

_VIT_MODEL: Optional[Any] = None
_VIT_PROCESSOR: Optional[Any] = None
_VIT_DEVICE: Optional[Any] = None
_VIT_LOAD_ATTEMPTED = False


def _normalize_label(raw_label: str) -> Optional[str]:
    """Normalize model label variants into the standard seven emotions."""
    cleaned = raw_label.strip().lower().replace("_", " ").replace("-", " ")
    label_map = {
        "anger": "angry",
        "angry": "angry",
        "disgust": "disgust",
        "disgusted": "disgust",
        "fear": "fear",
        "fearful": "fear",
        "happy": "happy",
        "happiness": "happy",
        "joy": "happy",
        "neutral": "neutral",
        "calm": "neutral",
        "sad": "sad",
        "sadness": "sad",
        "surprise": "surprise",
        "surprised": "surprise",
    }
    return label_map.get(cleaned)


def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(max(0.0, float(value)) for value in scores.values()))
    if total <= 1e-12:
        return {}
    normalized = {emotion: max(0.0, float(scores.get(emotion, 0.0))) / total for emotion in STANDARD_EMOTIONS}
    return normalized


def init_vit_emotion_model() -> bool:
    """Lazily initialize and cache the Hugging Face ViT emotion model."""
    global _VIT_MODEL, _VIT_PROCESSOR, _VIT_DEVICE, _VIT_LOAD_ATTEMPTED

    if _VIT_MODEL is not None and _VIT_PROCESSOR is not None and _VIT_DEVICE is not None:
        return True

    if _VIT_LOAD_ATTEMPTED:
        return False

    _VIT_LOAD_ATTEMPTED = True
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = AutoImageProcessor.from_pretrained(VIT_MODEL_NAME)
        model = AutoModelForImageClassification.from_pretrained(VIT_MODEL_NAME)
        model.to(device)
        model.eval()

        _VIT_MODEL = model
        _VIT_PROCESSOR = processor
        _VIT_DEVICE = device
        LOGGER.info("Loaded ViT emotion model '%s' on device=%s", VIT_MODEL_NAME, device)
        return True
    except Exception as exc:
        LOGGER.warning("Failed to initialize ViT emotion model: %s", exc)
        _VIT_MODEL = None
        _VIT_PROCESSOR = None
        _VIT_DEVICE = None
        return False


def _map_probs_to_standard(probabilities: Any) -> Dict[str, float]:
    scores = {emotion: 0.0 for emotion in STANDARD_EMOTIONS}

    if _VIT_MODEL is None:
        return {}

    id2label = getattr(getattr(_VIT_MODEL, "config", None), "id2label", {}) or {}
    for idx, prob in enumerate(list(probabilities)):
        raw_label = id2label.get(idx, id2label.get(str(idx), str(idx)))
        normalized = _normalize_label(str(raw_label))
        if normalized is None and idx < len(STANDARD_EMOTIONS):
            normalized = STANDARD_EMOTIONS[idx]
        if normalized is None:
            continue
        scores[normalized] += float(prob)

    return _normalize_scores(scores)


def detect_emotion_vit(face_crop: np.ndarray) -> Tuple[str, Dict[str, float]]:
    """Run ViT-based emotion inference on a BGR face crop.

    Args:
        face_crop: OpenCV BGR image of a face.

    Returns:
        (dominant_emotion, scores_dict). On any failure: ("neutral", {}).
    """
    if face_crop is None or getattr(face_crop, "size", 0) == 0:
        return "neutral", {}

    if not init_vit_emotion_model():
        return "neutral", {}

    try:
        import torch
        from PIL import Image

        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        if pil_image.size != (224, 224):
            pil_image = pil_image.resize((224, 224), Image.BILINEAR)

        processor = _VIT_PROCESSOR
        model = _VIT_MODEL
        device = _VIT_DEVICE
        if processor is None or model is None or device is None:
            return "neutral", {}

        inputs = processor(images=pil_image, return_tensors="pt")
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].detach().cpu().tolist()

        scores = _map_probs_to_standard(probabilities)
        if not scores:
            return "neutral", {}

        dominant_emotion = max(scores, key=scores.get)
        return dominant_emotion, scores
    except Exception as exc:
        LOGGER.warning("ViT emotion inference failed: %s", exc)
        return "neutral", {}


def benchmark_vit_emotion(face_crop: np.ndarray, runs: int = 10) -> Dict[str, float]:
    """Benchmark detect_emotion_vit and return average latency and FPS estimate."""
    if face_crop is None or getattr(face_crop, "size", 0) == 0:
        return {"avg_ms": 0.0, "fps_estimate": 0.0}

    total_runs = max(1, int(runs))
    total_seconds = 0.0

    # Warm up once so model loading and one-time setup do not skew average timing.
    detect_emotion_vit(face_crop)

    for _ in range(total_runs):
        start = time.perf_counter()
        detect_emotion_vit(face_crop)
        total_seconds += time.perf_counter() - start

    avg_seconds = total_seconds / float(total_runs)
    avg_ms = avg_seconds * 1000.0
    fps_estimate = (1.0 / avg_seconds) if avg_seconds > 0.0 else 0.0

    return {
        "avg_ms": round(avg_ms, 3),
        "fps_estimate": round(fps_estimate, 3),
    }
