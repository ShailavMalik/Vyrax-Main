#!/usr/bin/env python3
"""Validation Harness for Emotion Detection Pipeline.

Downloads RAF-DB and FER2013 validation splits, runs the full pipeline,
computes accuracy metrics, confusion matrix, and saves baseline report.
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, List

import cv2
import numpy as np
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Project imports
from emotion_engine import (
    EmotionIntelligenceEngine,
    get_face_detector_config,
    get_face_mesh_config,
    preprocess_face,
    detect_faces_mediapipe,
    extract_face,
    extract_features_mediapipe,
    detect_emotion_fer,
    detect_emotion_ensemble,
)
from config import SUPPORTED_EMOTIONS, FACE_CROP_SIZE

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

EMOTION_MAP = {
    'surprise': 0, 'fear': 1, 'disgust': 2, 'happy': 3,
    'sadness': 4, 'anger': 5, 'neutral': 6
}
EMOTION_LABELS = list(EMOTION_MAP.keys())

def emotion_to_idx(emotion: str) -> int:
    """Map emotion str to numeric index."""
    emotion = emotion.lower()
    return EMOTION_MAP.get(emotion, 6)  # default neutral

def load_raf_db_val_split(size: int = 1000) -> Tuple[List[str], List[int]]:
    """Load RAF-DB basic validation split (cropped faces)."""
    LOGGER.info(f'Loading RAF-DB val split (size={size})')
    # Note: RAF-DB needs manual download or use pre-cached.
    # Fall back to public FER2013 mirrors available on Hugging Face.
    dataset = None
    dataset_candidates = [
        ('fer2013', 'image', 'emotion'),
        ('clip-benchmark/wds_fer2013', 'jpg', 'cls'),
    ]
    split = f'test[:{size}]'
    for dataset_name, image_key, label_key in dataset_candidates:
        try:
            dataset = load_dataset(dataset_name, split=split)
            LOGGER.info(f'Loaded dataset={dataset_name} split={split}')
            break
        except Exception as exc:  # pragma: no cover - network/auth dependent
            LOGGER.warning(f'Failed dataset source {dataset_name}: {exc}')

    if dataset is None:
        raise RuntimeError(
            'Unable to load validation dataset. Checked: '
            + ', '.join(name for name, _, _ in dataset_candidates)
        )

    images = []
    labels = []
    for item in dataset:
        label = int(item[label_key])
        images.append(item[image_key])
        labels.append(label)
    return images, labels

def prepare_face_for_pipeline(image: Image.Image) -> np.ndarray:
    """Convert PIL to OpenCV, resize to detection frame size."""
    frame = np.array(image.convert('RGB'))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w = frame.shape[:2]
    if max(h, w) > 640:
        scale = 640 / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
    return frame

def run_pipeline_on_face(frame: np.ndarray, mode='baseline') -> str:
    """Run full emotion pipeline on single face frame.
    
    Args:
        frame: Input frame
        mode: 'baseline' or 'ensemble'
    """
    engine = EmotionIntelligenceEngine()
    
    # Detect face (should find the main face)
    detections = detect_faces_mediapipe(frame)
    if not detections:
        return 'uncertain'
    
    # Extract largest face
    sx1, sy1, sx2, sy2, _ = max(detections, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    face_crop = extract_face(frame, (sx1, sy1, sx2, sy2))
    if face_crop is None:
        return 'uncertain'
    
    # Preprocess and run FER
    norm_crop = preprocess_face(face_crop)
    if norm_crop is None:
        return 'uncertain'
    
    if mode == 'ensemble':
        _, scores, _ = detect_emotion_ensemble(norm_crop)
    else:
        emotion, scores = detect_emotion_fer(norm_crop)
        if emotion is None:
            return 'uncertain'
    
    # Features (for rules)
    mesh_config = get_face_mesh_config()
    features = extract_features_mediapipe(norm_crop, mesh_config)
    
    # Full engine eval
    now_ts = 0.0
    result = engine.evaluate(scores, features, now_ts)
    pred_emotion = result['final_emotion']
    
    return pred_emotion if result['final_confidence'] > 0.5 else 'uncertain'

def compute_metrics(gt_labels: List[int], pred_labels: List[int], labels: List[str]) -> Dict[str, Any]:
    """Compute full accuracy metrics."""
    # Filter uncertain
    mask = np.array(pred_labels) != len(labels)
    gt_f = np.array(gt_labels)[mask]
    pred_f = np.array(pred_labels)[mask]
    
    acc = accuracy_score(gt_f, pred_f)
    f1 = f1_score(gt_f, pred_f, average='weighted')
    
    cm = confusion_matrix(gt_f, pred_f)
    
    return {
        'accuracy': float(acc),
        'f1_weighted': float(f1),
        'confusion_matrix': cm.tolist(),
        'num_samples': len(gt_f)
    }

def plot_cm(cm: np.ndarray, labels: List[str], save_path: str):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Emotion Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main(args):
    """Main validation runner."""
    created_at = datetime.now(timezone.utc).isoformat()
    images, gt_labels = load_raf_db_val_split(args.size)
    
    pred_labels = []
    for img in tqdm(images, desc='Running pipeline'):
        frame = prepare_face_for_pipeline(img)
        pred = run_pipeline_on_face(frame)
        pred_idx = emotion_to_idx(pred)
        pred_labels.append(pred_idx)
    
    metrics = compute_metrics(gt_labels, pred_labels, EMOTION_LABELS)
    
    # Save report
    report = {
        'config': args.mode,
        'timestamp': created_at,
        'dataset_size': len(gt_labels),
        **metrics
    }
    
    report_path = Path('logs') / f'accuracy_report_{args.mode}.json'
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Confusion matrix
    cm_plot = report_path.with_suffix('.png')
    plot_cm(metrics['confusion_matrix'], EMOTION_LABELS, str(cm_plot))
    
    LOGGER.info(f"Accuracy: {metrics['accuracy']:.3f}")
    LOGGER.info(f"F1: {metrics['f1_weighted']:.3f}")
    LOGGER.info(f'Report saved: {report_path}')
    LOGGER.info(f'Matrix plot: {cm_plot}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1000, help='Dataset size')
    parser.add_argument('--mode', default='baseline', help='Run mode')
    args = parser.parse_args()
    main(args)
