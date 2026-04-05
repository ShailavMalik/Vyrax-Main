"""
Utility Functions for Image Processing and Box Operations

This module provides helper functions for:
- Frame resizing and preprocessing
- Bounding box manipulation
- Color and font utilities
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    """
    Resize a frame to a target width while preserving aspect ratio.
    
    Args:
        frame: Input image
        width: Target width in pixels
    
    Returns:
        Resized image
    """
    h, w = frame.shape[:2]
    if w <= 0 or w == width:
        return frame
    
    scale = width / float(w)
    new_h = int(h * scale)
    
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_LINEAR)


def clamp_box(x1: float, y1: float, x2: float, y2: float,
              frame_w: int, frame_h: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Clamp a bounding box to frame boundaries and validate it.
    
    This ensures the box stays within the image and is not degenerate
    (i.e., width and height > 0).
    
    Args:
        x1, y1, x2, y2: Box coordinates
        frame_w, frame_h: Frame dimensions
    
    Returns:
        Clamped box (x1, y1, x2, y2) or None if invalid
    """
    x1 = max(0, min(int(x1), frame_w - 1))
    y1 = max(0, min(int(y1), frame_h - 1))
    x2 = max(0, min(int(x2), frame_w - 1))
    y2 = max(0, min(int(y2), frame_h - 1))
    
    # Check for degenerate box
    if x2 <= x1 or y2 <= y1:
        return None
    
    return (x1, y1, x2, y2)


def add_padding_to_box(box: Tuple[int, int, int, int], padding: int,
                       frame_w: int, frame_h: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Expand a bounding box with padding without leaving the frame.
    
    Args:
        box: Bounding box (x1, y1, x2, y2)
        padding: Pixels to add on all sides
        frame_w, frame_h: Frame dimensions
    
    Returns:
        Padded and clamped box or None if invalid
    """
    x1, y1, x2, y2 = box
    return clamp_box(
        x1 - padding,
        y1 - padding,
        x2 + padding,
        y2 + padding,
        frame_w,
        frame_h
    )


def box_center_xyxy(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Get the center point of an xyxy bounding box.
    
    Args:
        box: Bounding box (x1, y1, x2, y2)
    
    Returns:
        Center point (cx, cy)
    """
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def box_width_height(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Get width and height of a bounding box.
    
    Args:
        box: Bounding box (x1, y1, x2, y2)
    
    Returns:
        Tuple of (width, height)
    """
    x1, y1, x2, y2 = box
    return (x2 - x1, y2 - y1)


def draw_text_with_bg(frame: np.ndarray, text: str, x: int, y: int,
                      font: int, font_scale: float, text_color: Tuple[int, int, int],
                      thickness: int, bg_color: Tuple[int, int, int],
                      padding: int = 6) -> None:
    """
    Draw text on a frame with a solid background rectangle.
    
    This improves text readability against variable backgrounds.
    
    Args:
        frame: Input/output image
        text: Text to draw
        x, y: Text position (bottom-left corner)
        font: OpenCV font face
        font_scale: Font size multiplier
        text_color: Text color (BGR)
        thickness: Text thickness
        bg_color: Background color (BGR)
        padding: Pixels to pad around text
    """
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_w, text_h = text_size
    
    # Draw background rectangle
    cv2.rectangle(
        frame,
        (x - padding, y - text_h - padding),
        (x + text_w + padding, y + padding),
        bg_color,
        -1,  # -1 means filled
    )
    
    # Draw text on top
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def is_box_overlapping(box1: Tuple[int, int, int, int],
                       box2: Tuple[int, int, int, int],
                       overlap_threshold: float = 0.1) -> bool:
    """
    Check if two bounding boxes overlap significantly.
    
    Args:
        box1, box2: Bounding boxes (x1, y1, x2, y2)
        overlap_threshold: Minimum IoU to consider overlapping
    
    Returns:
        True if boxes overlap above threshold
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    if xi2 < xi1 or yi2 < yi1:
        return False  # No intersection
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    
    return iou >= overlap_threshold