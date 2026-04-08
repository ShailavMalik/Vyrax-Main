# Emotion Detection Models and Rule Intelligence (Detailed Technical Brief)

This document explains the exact model stack used in this project, how the runtime pipeline works, how MediaPipe-based rules improve results, and the small implementation details that reduce flicker and false positives.

Use this file as context when asking ChatGPT for suggestions about improving detection quality, rules, thresholds, or robustness.

## Purpose Of This File (Plain English)

This file exists to help ChatGPT act like a technical advisor for your project.

You are currently using FER + MediaPipe mesh + custom rule logic. The goal is:

1. Improve detection quality for all emotions.
2. Reduce flicker and wrong emotion switches.
3. Avoid manual trial-and-error after every small threshold change.
4. Get repeatable suggestions based on benchmark numbers, not only visual guesswork.

In short: this document gives ChatGPT enough context so it can recommend better rules and tuning with less back-and-forth from you.

---

## 1. What Models Are Used

### 1.1 Face Detection Model (for locating face box)

The system uses **MediaPipe face detection** through two possible runtime backends:

1. **Primary path (MediaPipe Solutions API):**
   - `solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.25)`
   - `solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.20)`
   - Both are attempted; detections are combined and sorted by confidence.

2. **Fallback path (MediaPipe Tasks API):**
   - Model file: `models/face_detector.tflite`
   - Source URL in code points to `blaze_face_short_range`.

So, practically, face detection is MediaPipe BlazeFace-family behavior, with short/full range paths depending on backend availability.

### 1.2 Face Landmark Model (for geometry/rules)

The system uses **MediaPipe FaceMesh / FaceLandmarker**:

1. **Primary path (Solutions API):** `solutions.face_mesh.FaceMesh(...)`
2. **Fallback path (Tasks API):** `models/face_landmarker.task`

This is used to compute geometry-based features such as mouth opening, eye opening, eyebrow movement, smile shape, and expression intensity.

### 1.3 Emotion Classification Model (FER)

Emotion inference uses a **two-level strategy**:

1. **Primary classifier:** `fer.FER(mtcnn=False)`
   - Called in `detect_emotion_fer(...)`
   - Returns per-emotion score dictionary.

2. **Fallback classifier:** `DeepFace.analyze(..., actions=["emotion"], enforce_detection=False, silent=True)`
   - Used only if FER is unavailable or fails.

Important: the repository code does not hardcode a custom checkpoint path for FER/DeepFace emotion models. It relies on the models shipped/managed by those libraries in the active Python environment.

---

## 2. End-to-End Runtime Flow

### 2.1 Frame Scheduling Strategy

The pipeline does **not** run heavy emotion inference on every frame. It uses trigger-driven scheduling:

Heavy inference runs when at least one is true:

- Time trigger: elapsed time exceeds `HEAVY_DETECTION_INTERVAL_SECONDS` (default 2.0s)
- Face movement trigger: box center shift exceeds `FACE_MOVEMENT_THRESHOLD_PX` (default 18 px)
- Landmark change trigger: signature delta exceeds `LANDMARK_CHANGE_THRESHOLD` (default 0.07)

Otherwise, cached results are reused in tracking mode.

### 2.2 Main Processing Stages

For a heavy frame:

1. Detect face on a small resized frame (`DETECTION_FRAME_SIZE` = 320x240)
2. Scale the box back to original frame coordinates
3. Crop face with padding and resize
4. Preprocess face (CLAHE in LAB then YCrCb)
5. Run FER (or fallback DeepFace)
6. Extract MediaPipe landmark features
7. Compute face-quality score (sharpness + brightness)
8. Run intelligence engine:
   - calibration
   - quality-aware adjustment
   - low-confidence filtering
   - emotion weighting
   - facial rule correction
   - temporal smoothing
   - transition gating
   - hold logic
   - optional raw override
   - IoT actuation debouncing
9. Cache result for tracking frames and render output

---

## 3. Score Processing Pipeline (EmotionIntelligenceEngine)

Given raw FER scores, the intelligence layer performs the following sequence.

### 3.1 Normalization

Scores are normalized to a probability-like distribution and aligned to supported emotions:

- angry, disgust, fear, happy, sad, surprise, neutral

### 3.2 Confidence Calibration

A power calibration is applied:

- `score_calibrated = score ^ CONFIDENCE_CALIBRATION_POWER`
- Default power: `0.92`

This gently reshapes probabilities before later logic.

### 3.3 Face Quality Compensation

Quality is estimated from:

- Laplacian variance (sharpness)
- Brightness closeness to mid-range exposure

If quality is low (`q < 0.55`):

- non-neutral classes are penalized
- neutral is boosted slightly

Purpose: reduce noisy confident misclassifications on blurry/poor-lit frames.

### 3.4 Low-Confidence Uncertainty Filter

Uses:

- `MIN_CONFIDENCE_THRESHOLD` (default 0.12)
- `LOW_CONFIDENCE_GAP_THRESHOLD` (default 0.08)

If top class confidence is too low OR top1-top2 gap is too small:

- mark uncertain
- reduce overconfident winner slightly
- keep distribution explainable

### 3.5 Emotion Prior Weights

Per-emotion prior multipliers are applied (then renormalized):

- happy 1.02
- sad 1.08
- angry 1.20
- surprise 1.30
- fear 1.08
- disgust 1.40
- neutral 0.90

Purpose: compensate common base-model bias and improve rare-expression visibility.

### 3.6 Facial Rule Engine (MediaPipe geometry)

Rule engine uses landmark-derived metrics:

- `mouth_open_ratio`
- `lip_spread_ratio`
- `eye_open_ratio`
- `eyebrow_position`
- `eyebrow_inner_raise`
- `smile_ratio`
- `expression_intensity`

Rules are adaptive: thresholds are shifted using a per-user EMA baseline (`FEATURE_BASELINE_ALPHA=0.92`).

### 3.7 Temporal Smoothing

Two layers:

1. Recency-weighted window smoothing (`SMOOTHING_WINDOW_SIZE=8`)
2. Optional EMA smoothing (`USE_EMA_SMOOTHING=True`, `EMA_ALPHA=0.40`)

### 3.8 Transition Control

Prevents rapid oscillation with:

- transition margin (`TRANSITION_MARGIN=0.09`)
- cooldown (`TRANSITION_COOLDOWN_SECONDS=0.35`)
- candidate streak logic (needs repeated wins for easy switching)
- strong-candidate fast path

### 3.9 Hold Logic

Emotion labels are held for stability:

- `EMOTION_HOLD_SECONDS=2.0`
- requires margin/streak for switching during hold
- strong happy/surprise can escape hold via dedicated thresholds

### 3.10 Raw Confidence Override

If raw FER is very confident and consistent:

- confidence >= `RAW_OVERRIDE_CONFIDENCE` (0.60)
- streak >= `RAW_OVERRIDE_STREAK` (2)
- outperforms final by `RAW_OVERRIDE_MARGIN` (0.10)

Then raw label can override post-processed label to avoid over-smoothed drift.

### 3.11 IoT Actuation Stabilizer (separate output)

The engine also computes a debounced signal for actuators:

- minimum confidence gate
- uncertain blocking
- confirmation streak
- hold timer for command stability

This is independent from the on-screen label, so control systems receive a steadier signal.

---

## 4. MediaPipe Rule Logic: How It Improves Results

### 4.1 Neutral Guard for Weak Expressions

If expression intensity is low and neutral already has support:

- boost neutral
- suppress sad/angry slightly

Why: prevents false sad/angry predictions on nearly neutral faces.

### 4.2 Surprise Boost (primary correction)

Strong surprise patterns are promoted when geometry indicates:

- open mouth
- raised inner brows
- optional extra support from wide eyes and high intensity

Secondary surprise cue also boosts surprise when mouth+eyes are open and surprise/fear prior is plausible.

Why: FER can under-detect surprise in transient frames; geometry is very informative here.

### 4.3 Angry Boost

Promoted when:

- lowered brows
- narrowed eyes
- mouth not too open
- angry support is plausible and not dominated by sad

Why: improves angry recognition when FER wavers between angry/sad/neutral.

### 4.4 Sad Boost

Promoted when:

- raised inner brows
- narrowed eyes
- non-angry brow geometry
- sad-like mouth shape (low smile, low lip spread)

Why: separates sadness from anger and neutral in subtle cases.

### 4.5 Happy Override

If smile or lip spread is strong and happy already has enough support:

- add a strong happy boost

Why: helps recover smile-driven happy when classifier confidence is diluted.

### 4.6 Adaptive Baseline (personalization)

Each user has a rolling facial baseline from recent landmarks.
Thresholds are adjusted relative to this baseline.

Why: fixed absolute geometry thresholds are brittle across different face shapes and camera distances.

---

## 5. Minor But Important Quality Improvements

These details significantly improve practical stability:

1. Multi-pass face detection:

- original frame
- contrast-enhanced frame
- upscaled enhanced frame

2. CLAHE preprocessing before FER:

- reduces sensitivity to lighting changes

3. Detection on low resolution, inference on original crop:

- saves compute while preserving emotion quality

4. Trigger-based heavy inference:

- less compute and less noisy frame-to-frame overreaction

5. Box smoothing and short missing-box hold:

- reduces UI jitter and blinking

6. Rule trigger logging in JSONL:

- gives explainability and offline analysis for threshold tuning

7. Benchmark reports:

- allows quantitative comparisons of tuning changes

---

## 6. Current Tunable Knobs (Most Impactful)

If asking ChatGPT for suggestions, focus first on these:

1. Detection/trigger cadence:

- `HEAVY_DETECTION_INTERVAL_SECONDS`
- `FACE_MOVEMENT_THRESHOLD_PX`
- `LANDMARK_CHANGE_THRESHOLD`

2. Uncertainty behavior:

- `MIN_CONFIDENCE_THRESHOLD`
- `LOW_CONFIDENCE_GAP_THRESHOLD`

3. Stability/latency tradeoff:

- `SMOOTHING_WINDOW_SIZE`
- `EMA_ALPHA`
- `TRANSITION_MARGIN`
- `EMOTION_HOLD_SECONDS`

4. Rule sensitivity:

- mouth/eye/brow/smile thresholds
- `RULE_MIN_*` guards
- surprise/angry/sad boost constants

5. Quality compensation:

- `LOW_QUALITY_NON_NEUTRAL_PENALTY`
- `LOW_QUALITY_NEUTRAL_BOOST`

6. Emotion priors:

- `EMOTION_WEIGHTS`

---

## 7. Known Limitations to Share with ChatGPT

1. Single-face processing default (`MAX_FACES_PROCESS=1`):

- optimized for one dominant face

2. Rule set is heuristic:

- transparent and tunable, but may need dataset-driven calibration

3. FER/DeepFace model internals are external:

- behavior can vary by installed library versions

4. Landmark metrics are 2D image-space:

- depth/pose extremes may still cause edge-case errors

5. No supervised meta-classifier yet:

- current correction stack is deterministic + temporal logic

---

## 8. Suggested Prompt Template for ChatGPT

Use this template when requesting improvement advice:

```
I have a real-time webcam emotion pipeline with:
- Face detection: MediaPipe BlazeFace paths
- Landmarks: MediaPipe FaceMesh/FaceLandmarker
- Emotion classifier: fer.FER(mtcnn=False), DeepFace emotion fallback
- Post-processing: calibration, quality compensation, low-confidence filtering,
  emotion priors, MediaPipe geometry rules, smoothing, transition gating, hold,
  and raw-override.

Current objective:
[Describe exact issue, e.g., surprise is delayed, sad/neutral confusion, flicker]

Current constraints:
- Real-time CPU target
- Must remain interpretable
- Keep deterministic rule layer (no full retrain)

Please propose:
1) threshold/rule changes
2) new geometric features
3) validation protocol and metrics
4) low-risk rollout sequence
```

### 8.1 Better Prompt For "Improve All Emotions" (Low Manual Testing)

Use this when you want ChatGPT to propose changes that generalize better, with minimal manual verification:

```text
You are helping tune a real-time FER + MediaPipe rule engine.

Context:
- Base emotion model: FER (DeepFace fallback)
- Geometry: MediaPipe FaceMesh features
- Post-processing: calibration, quality compensation, uncertainty filter,
  weighted priors, adaptive geometric rules, smoothing, transition gating, hold, raw override

Your task:
1) Propose a balanced update for all 7 emotions (angry, disgust, fear, happy, sad, surprise, neutral).
2) Do not optimize for one emotion only.
3) Minimize flicker and neutral confusion.
4) Keep CPU-friendly real-time behavior.

Return output in 4 blocks:
BLOCK A: Parameter changes (exact constants and old->new values)
BLOCK B: Rule changes (if/then logic updates)
BLOCK C: Risk analysis (what might regress)
BLOCK D: Validation checklist with pass/fail thresholds
```

### 8.2 What To Ask ChatGPT To Avoid Overfitting

Ask it to optimize this objective function, not a single metric:

- Higher average final confidence
- Lower uncertain rate
- Lower emotion switch rate
- Stable or improved heavy-rate/FPS

This encourages globally useful tuning instead of one-class hacks.

---

## 10. "No Manual Re-Testing" Workflow

Use this loop so you do not need to manually inspect every tweak.

### 10.1 Establish a Baseline Once

Run and save a baseline report:

```bash
python realtime_emotion.py --benchmark-seconds 60 --benchmark-report logs/benchmark_baseline_60s.json
```

Keep this baseline fixed while you evaluate future rule changes.

### 10.2 Evaluate Each Proposed Change With the Same Protocol

For every ChatGPT suggestion, run the same benchmark duration and save a new report:

```bash
python realtime_emotion.py --benchmark-seconds 60 --benchmark-report logs/benchmark_candidate_60s.json
```

### 10.3 Use Simple Acceptance Gates

Only keep a change if all major gates pass:

1. `uncertain_rate` does not increase.
2. `emotion_switches` does not increase significantly.
3. `avg_final_confidence` improves or stays stable.
4. `fps_effective` does not drop below your acceptable threshold.

If one gate fails, reject or partially roll back that suggestion.

### 10.4 Ask ChatGPT To Compare Baseline vs Candidate

Paste both JSON reports and ask:

```text
Compare these two benchmark reports.
Decide KEEP/REJECT for this tuning patch.
Explain metric-by-metric tradeoffs.
If REJECT, propose the smallest safe adjustment and why.
```

### 10.5 Monthly Rule Consolidation

Instead of tuning every day, batch multiple suggestions and do one consolidation pass:

1. Collect 3-5 candidate tweaks.
2. Benchmark each with same protocol.
3. Keep only top 1-2 that pass gates.
4. Re-baseline.

This avoids endless micro-adjustment cycles.

---

## 9. File Pointers (for direct code inspection)

- `emotion_engine.py`
  - model init and detection helpers
  - `detect_emotion_fer(...)`
  - `extract_features_mediapipe(...)`
  - `apply_facial_rules(...)`
  - `EmotionIntelligenceEngine`
- `realtime_emotion.py`
  - trigger-driven scheduling
  - heavy vs tracking frame logic
  - rendering and benchmark/report integration
- `config.py`
  - thresholds, weights, smoothing, hold, and quality constants

---

This document reflects the current implementation and is designed to be pasted directly into ChatGPT context for improvement-oriented discussions.
