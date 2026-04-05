# Real-Time Emotion Detection System

A **production-style**, **fully modular**, and **well-documented** real-time emotion detection system using OpenCV, FER/DeepFace, and MediaPipe.

## 🎯 Architecture Overview

The system is built across **6 distinct phases**, each with clear responsibilities:

```
WEBCAM FEED
    ↓
┌─────────────────────────────────────────────────────┐
│ PHASE 1: BASE EMOTION DETECTION (FER)               │
│ - Capture frames with OpenCV                        │
│ - Detect faces with RetinaFace                      │
│ - Analyze emotions with DeepFace (FER model)        │
│ - Extract confidence scores for all emotions        │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ PHASE 2: EMOTION ENGINE (LOGIC LAYER)               │
│ - Maintain rolling buffer (last N frames)           │
│ - Filter low-confidence predictions                 │
│ - Apply majority voting smoothing                   │
│ - Prevent rapid emotion switching                   │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ PHASE 3: MEDIAPIPE FEATURE EXTRACTION               │
│ - Extract facial landmarks                          │
│ - Compute mouth openness                            │
│ - Compute eye openness                              │
│ - Compute eyebrow raise                             │
│ - Estimate head tilt & gaze direction               │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ PHASE 4: RULE-BASED CORRECTION                      │
│ - Apply logic rules:                                │
│   • neutral + wide mouth + eyes → surprise          │
│   • raised brows + wide eyes → fear/surprise        │
│ - Combine FER with MediaPipe features               │
│ - Apply emotion weights                             │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ PHASE 5: FINAL PIPELINE                             │
│ - Integrate all 4 phases                            │
│ - Real-time frame rendering                         │
│ - Display confidence & metrics                      │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ PHASE 6: PERFORMANCE OPTIMIZATION                   │
│ - Frame skipping (process every Nth frame)          │
│ - Result caching                                    │
│ - Single-face processing                            │
│ - FPS limiting                                      │
└─────────────────────────────────────────────────────┘
    ↓
  DISPLAY (OpenCV Window)
```

## 📁 File Structure

```
deepface-demo/
├── config.py                  # Central configuration & constants
├── detection.py              # PHASE 1 & 2: Face detection & emotion engine
├── optional_features.py      # PHASE 3: MediaPipe feature extraction
├── utils.py                  # Helper utilities & image processing
├── rendering.py              # Visualization & drawing utilities
├── tracking.py               # Face tracking across frames
├── realtime_emotion.py       # PHASE 5 & 6: Main pipeline & entry point
├── models/
│   └── face_landmarker.task  # MediaPipe FaceLandmarker model
└── README.md                 # This file
```

## 🗂️ Module Guide

### `config.py` - Configuration & Constants

Central hub for all settings organized by phase:

- Camera & frame settings
- Smoothing & confidence thresholds
- MediaPipe feature thresholds
- Emotion weights
- Visual rendering colors
- Performance optimization parameters

**Example:**

```python
SMOOTHING_WINDOW_SIZE = 8       # Frames to keep in rolling buffer
CONFIDENCE_THRESHOLD = 0.15     # Min confidence to accept prediction
PROCESS_EVERY_N_FRAMES = 3      # Skip frames for performance
EMOTION_WEIGHTS = {...}         # Boost/reduce certain emotions
```

### `detection.py` - Face Detection & Emotion Engine

Implements PHASE 1 and PHASE 2:

- `detect_faces()` - RetinaFace face detection
- `detect_emotion_fer()` - DeepFace emotion analysis
- `crop_face_region()` - Extract padded face images
- `EmotionHistory` class - Rolling buffer with smoothing
- `apply_emotion_weights()` - Correct FER bias

**Key Class: `EmotionHistory`**

```python
history = EmotionHistory(window_size=8)
history.add_prediction("happy", 0.92)
smoothed, agreement = history.get_smoothed_emotion()  # "happy", 0.85
```

### `optional_features.py` - MediaPipe Features

Implements PHASE 3:

- `init_mediapipe_facemesh()` - Initialize MP backend
- `extract_features_mediapipe()` - Extract facial features
- `has_open_mouth()`, `has_wide_eyes()`, `has_raised_eyebrows()` - Feature checks
- Mouth openness, eye openness, eyebrow raise detection

**Features Extracted:**

```python
features = extract_features_mediapipe(frame, config)
# Returns:
# {
#   "mouth_openness": 0.0-1.0,
#   "eye_openness": 0.0-1.0,
#   "eyebrow_raise": 0.0-1.0,
#   "head_tilt": -1.0 to 1.0,
#   "gaze_direction": "left" | "center" | "right"
# }
```

### `utils.py` - Helper Utilities

Common functions for all modules:

- `resize_to_width()` - Maintain aspect ratio resizing
- `clamp_box()` - Keep boxes within frame boundaries
- `draw_text_with_bg()` - Draw readable text with background
- `box_center_xyxy()` - Get box center point

### `rendering.py` - Visualization

Functions for drawing results:

- `draw_face_box()` - Draw bounding box + emotion label
- `draw_multi_face_results()` - Draw results for multiple faces
- `draw_performance_metrics()` - Display FPS & timing

### `tracking.py` - Face Tracking

Track faces across frames:

- `FaceTracker` class - Centroid-based tracking
- `add_emotion_to_track()` - Add emotion history to track
- `get_track_emotion_summary()` - Get smoothed emotion from track

### `realtime_emotion.py` - Main Pipeline

Implements PHASE 4, 5, and 6:

- `apply_emotion_rules()` - Rule-based corrections (PHASE 4)
- `EmotionDetectionPipeline` class - Main orchestration
- `main()` - Entry point

**Main Pipeline Class:**

```python
pipeline = EmotionDetectionPipeline()
pipeline.initialize_camera()
pipeline.run()  # Main loop - processes frames until 'q' pressed
```

## ✨ Key Features

### **Modular Design**

- Each phase is independent and testable
- Functions are focused and reusable
- Clean separation of concerns

### **Comprehensive Documentation**

- Docstrings for all functions & classes
- Inline comments for complex logic
- Type hints throughout

### **Performance Optimization (PHASE 6)**

- Process every Nth frame (configurable)
- Cache results for skipped frames
- Single-face processing limit
- Real-time FPS display

### **Robust Error Handling**

- Graceful handling of detection failures
- Silent exceptions with fallbacks
- No system crashes on bad data

### **Easy to Extend**

- Add new rule-based corrections easily
- Modify emotion weights per use case
- Adjust thresholds in config.py
- Integrate with FastAPI or other frameworks

## 🚀 Quick Start

### Prerequisites

```bash
pip install opencv-python deepface tf-keras mediapipe retina-face
```

### Run the System

```bash
python realtime_emotion.py
```

### Run With Benchmark Mode

```bash
python realtime_emotion.py --benchmark-seconds 30 --benchmark-report logs/benchmark_report.json
```

This runs the live pipeline for a fixed time and emits a JSON summary report with:

- heavy-vs-tracking frame rate
- trigger source counts (time, movement, landmark)
- uncertain prediction rate
- average raw/final confidence
- effective FPS

### Structured Event Logs

Heavy inference events are saved to:

```text
logs/emotion_events.jsonl
```

Each event includes raw FER scores, rule triggers, final emotion/confidence, trigger source, and FPS.

### Controls

- **`q` key**: Quit the application
- **Window**: Live preview with emotion labels

### Expected Output

```
========================================================
REAL-TIME EMOTION DETECTION SYSTEM
========================================================
Press 'q' to quit
========================================================

✓ Camera initialized

[Face 1]
- Emotion: HAPPY (92%)
- Confidence: 0.92
- FPS: 28.5

[Face 2]
- Emotion: NEUTRAL (68%)
- Confidence: 0.68
```

## ⚙️ Configuration

All settings are in `config.py`. Some important parameters:

```python
# Smoothing
SMOOTHING_WINDOW_SIZE = 8
CONFIDENCE_THRESHOLD = 0.15

# Performance
PROCESS_EVERY_N_FRAMES = 3    # Process every 3rd frame
MAX_FACES_PROCESS = 1          # Only analyze strongest face

# Display
SHOW_FPS = True
SHOW_CONFIDENCE = True
SHOW_RAW_FER_SCORE = True

# Emotion weights
EMOTION_WEIGHTS = {
    "happy": 1.15,      # Boost happy
    "surprise": 1.25,   # Strongly boost surprise
    "neutral": 0.85,    # Reduce neutral dominance
}
```

## 🔧 Common Customizations

### Modify Emotion Rules (PHASE 4)

Edit `realtime_emotion.py`, function `apply_emotion_rules()`:

```python
def apply_emotion_rules(emotion, features, scores):
    # Add your custom rules here
    if emotion == "neutral" and has_open_mouth(features):
        return "surprise", new_confidence
    return emotion, confidence
```

### Adjust Smoothing

```python
# In config.py
SMOOTHING_WINDOW_SIZE = 15  # Increase for more smoothing
CONFIDENCE_THRESHOLD = 0.25  # Higher = filter more predictions
```

### Change Detection Speed

```python
# In config.py
PROCESS_EVERY_N_FRAMES = 1   # Process every frame (slower but more responsive)
PROCESS_EVERY_N_FRAMES = 5   # Skip more frames (faster but less responsive)
```

## 📊 Understanding Confidence Scores

```
RAW FER SCORE
    ↓ (0-1, from DeepFace)
WEIGHTED              (Apply EMOTION_WEIGHTS)
    ↓
SMOOTHED            (Majority vote over last 8 frames)
    ↓ (0-1, agreement ratio)
FINAL CONFIDENCE    (Displayed to user)
```

The confidence represents **how many recent frames agree** on the emotion, not the model's internal confidence.

## 🐛 Troubleshooting

### No faces detected

- Ensure good lighting
- Position face clearly in view
- Check camera is working: `python -c "import cv2; cap = cv2.VideoCapture(0)"`

### Jerky emotion changes

- Increase `SMOOTHING_WINDOW_SIZE` in config.py
- Increase `CONFIDENCE_THRESHOLD`

### Slow FPS

- Increase `PROCESS_EVERY_N_FRAMES`
- Reduce `FRAME_WIDTH`
- Disable MediaPipe feature extraction (optional)

### Wrong emotions detected

- Adjust `EMOTION_WEIGHTS` in config.py
- Modify rules in `apply_emotion_rules()`
- Retrain with your own data

## 📚 Learning Path

1. **Understand the flow**: Read this README
2. **Study Phase 1**: `detection.py` - face detection & FER
3. **Study Phase 2**: `EmotionHistory` class in `detection.py`
4. **Study Phase 3**: `optional_features.py` - MediaPipe landmarks
5. **Study Phase 4**: `apply_emotion_rules()` in `realtime_emotion.py`
6. **Study Phase 5**: `EmotionDetectionPipeline` class structure
7. **Study Phase 6**: Frame skipping & caching logic
8. **Experiment**: Modify config & rules for your use case

## 🎓 Educational Value

This codebase demonstrates:

- Clean architecture principles
- Modular design patterns
- Type hints & documentation best practices
- Real-time processing optimization
- Computer vision fundamentals
- State management (emotion history)
- Error handling strategies

## 📖 Dependencies

| Package         | Purpose                    |
| --------------- | -------------------------- |
| `opencv-python` | Video capture & rendering  |
| `deepface`      | FER emotion detection      |
| `mediapipe`     | Facial landmark extraction |
| `tf-keras`      | Underlying neural networks |
| `retina-face`   | Face detection             |
| `numpy`         | Numerical operations       |

## 🔮 Future Enhancements

- [ ] FastAPI server for remote emotion detection
- [ ] Database logging of emotions over time
- [ ] Multi-face tracking with IDs
- [ ] Emotion transition analysis
- [ ] Custom model training
- [ ] GPU acceleration (CUDA)
- [ ] Real-time emotion statistics dashboard
- [ ] Export to video file with annotations

## 📝 Notes

- The system processes one face at a time (most prominent)
- Emotions are from 7-class model: angry, disgust, fear, happy, neutral, sad, surprise
- All coordinates are in (x1, y1, x2, y2) format (xyxy)
- Colors are in BGR format (OpenCV standard)
- Confidence ranges from 0.0 (no agreement) to 1.0 (full agreement)

## 📄 License

This project is provided as-is for educational and development purposes.

---

**Built with ❤️ for real-time emotion detection**

Questions or suggestions? Feel free to modify and extend!
