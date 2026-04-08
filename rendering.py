"""
RENDERING MODULE: Display Real-Time Emotion Data

This module handles all visual output:
- Bounding boxes around detected faces
- Emotion labels with confidence scores
- Performance metrics (FPS)
- Feature visualization (optional)
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List, Any

from config import (
    BOX_THICKNESS,
    BG_COLOR,
    BG_PADDING,
    DEFAULT_EMOTION_COLOR,
    EMOTION_COLORS,
    FONT,
    FONT_SCALE,
    TEXT_COLOR,
    SHOW_FPS,
)
from utils import draw_text_with_bg


# ============================================================================
# BOUNDING BOX AND LABEL DRAWING
# ============================================================================

def draw_face_box(frame: np.ndarray, box: Tuple[int, int, int, int],
                  emotion: str, confidence: float,
                  raw_emotion: Optional[str] = None,
                  raw_confidence: Optional[float] = None) -> None:
    """
    Draw a face bounding box with emotion label and confidence.
    
    Args:
        frame: Image to draw on (modified in-place)
        box: Bounding box (x1, y1, x2, y2)
        emotion: Final emotion prediction (str)
        confidence: Confidence score (0-1)
        raw_emotion: Raw FER emotion (optional, for display)
        raw_confidence: Raw FER confidence (optional)
    """
    x1, y1, x2, y2 = box
    
    # Get color for this emotion
    color = EMOTION_COLORS.get(emotion, DEFAULT_EMOTION_COLOR)
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
    
    # Prepare emotion label
    label = f"{emotion.upper()}: {confidence*100:.0f}%"
    
    # Draw label with background
    draw_text_with_bg(
        frame,
        label,
        x1 + 10,
        y1 - 10,
        FONT,
        FONT_SCALE,
        TEXT_COLOR,
        2,
        color,
        BG_PADDING
    )


def draw_multi_face_results(frame: np.ndarray,
                            faces: List[Dict[str, Any]]) -> None:
    """
    Draw results for multiple detected faces.
    
    Args:
        frame: Image to draw on
        faces: List of face dicts with keys:
               - box: (x1, y1, x2, y2)
               - emotion: detected emotion
               - confidence: confidence score
               - raw_emotion: (optional) raw FER emotion
               - raw_confidence: (optional) raw FER confidence
    """
    for face in faces:
        draw_face_box(
            frame,
            face["box"],
            face["emotion"],
            face["confidence"],
            face.get("raw_emotion"),
            face.get("raw_confidence")
        )


def draw_performance_metrics(frame: np.ndarray, fps: float,
                             process_time: float) -> None:
    """
    Draw FPS and processing time on the frame.
    
    Args:
        frame: Image to draw on
        fps: Frames per second
        process_time: Time to process last frame (milliseconds)
    """
    if not SHOW_FPS:
        return
    
    metrics_text = f"FPS: {fps:.1f} | Time: {process_time:.1f}ms"
    
    draw_text_with_bg(
        frame,
        metrics_text,
        10,
        30,
        FONT,
        FONT_SCALE * 0.8,
        (0, 255, 0),  # Green
        1,
        BG_COLOR,
        BG_PADDING
    )
