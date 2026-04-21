# Vyra-X: Real-Time Emotion Insight Platform

Vyra-X is a real-time emotion intelligence system that combines webcam inference, live UI visualization, and cloud-backed snapshot/event storage for analytics and review.

The project is designed for live demos and judging: it provides an immediate visual experience, transparent technical pipeline, and traceable data in MongoDB plus Azure Blob Storage.

## What Problem It Solves

Vyra-X turns raw facial video into structured, time-aware emotional insights. It is useful in scenarios where teams need both live understanding and historical evidence:

- Human-computer interaction demos and research.
- Classroom/training feedback (engagement, confusion trends).
- Interview/practice analysis sessions.
- Usability testing and emotional response tracking.

## Key Features

- Real-time webcam emotion inference with stabilized output.
- Live MJPEG video stream with emotion overlays.
- Adaptive snapshot gating to avoid noisy over-capture.
- Cloud snapshot persistence to Azure Blob Storage.
- Analytics-ready metadata in MongoDB collections.
- Session-level summary generation (dominant emotion, confidence, event counts).
- Admin-friendly APIs for snapshots, emotion events, and summaries.

## System Architecture

### Frontend (React + Vite)

- Renders live camera feed and emotion state.
- Polls backend endpoints for current inference output.
- Supports admin visualizations for snapshots/events/summary.

### Backend (Flask + CV/ML Pipeline)

- Captures frames from webcam.
- Runs emotion detection and smoothing pipeline.
- Streams annotated frames over MJPEG.
- Emits current emotion state as JSON.
- Persists selected snapshots and events.

### Data Layer

- Azure Blob Storage: image binary storage.
- MongoDB:
	- snapshots collection: snapshot metadata.
	- emotions collection: per-event emotional timeline.
	- sessions collection: computed per-session summaries.

## Repository Structure

```text
vyrax-huggingface/
├─ backend/
│  ├─ web_app.py
│  ├─ admin_store.py
│  ├─ snapshot_storage.py
│  ├─ realtime_emotion.py
│  ├─ emotion_engine.py
│  ├─ config.py
│  ├─ models/
│  ├─ logs/
│  └─ requirements.txt
├─ frontend/
│  ├─ src/
│  ├─ public/
│  └─ package.json
├─ start-all.ps1
├─ .env
└─ README.md
```

## End-to-End Flow

1. Frame is read from webcam.
2. Inference pipeline computes emotion and confidence.
3. Backend updates live state for UI endpoints.
4. Snapshot gate decides whether to store image:
	 - First snapshot in a session.
	 - Emotion change after cooldown.
	 - Periodic high-confidence checkpoint.
	 - Periodic keepalive checkpoint.
5. If accepted:
	 - Image uploaded to Azure Blob Storage.
	 - Metadata written to MongoDB snapshots.
6. Emotion events are written to MongoDB emotions.
7. Session summary is updated in MongoDB sessions.

## Snapshot Timing and Gating

Snapshot capture is event-based, not fixed-interval only. Current demo profile from .env:

- SNAPSHOT_MIN_INTERVAL_SECONDS=20
- SNAPSHOT_HIGH_CONFIDENCE_INTERVAL_SECONDS=75
- SNAPSHOT_KEEPALIVE_INTERVAL_SECONDS=240
- SNAPSHOT_HIGH_CONFIDENCE=0.85

Interpretation:

- First snapshot of a session is stored immediately.
- If emotion changes, at least 20 seconds must pass from the previous snapshot.
- If emotion is stable but confidence is high, store every 75 seconds.
- Otherwise store keepalive snapshots every 240 seconds.

Additional quality gates:

- Dark frame rejection using SNAPSHOT_MIN_LUMA.
- Low face-quality rejection using SNAPSHOT_MIN_FACE_QUALITY.

## Performance Tuning for Judge Demo

This repository is currently tuned for smooth live demonstration:

- SNAPSHOT_CLEANUP_EVERY_UPLOADS=20
	- Cleanup runs once every 20 uploads (reduced per-upload overhead).
- MJPEG_QUALITY=65
	- Reduces CPU/network pressure while maintaining clear stream quality.
- Longer periodic snapshot intervals
	- Reduces storage write churn and backend contention.

If your hardware is strong and you want denser analytics, reduce intervals and increase MJPEG quality. If hardware is constrained, do the opposite.

## Environment Variables

Create a root-level .env file with these values:

```env
# Azure
AZURE_BLOB_CONNECTION_STRING=<your_connection_string>
AZURE_BLOB_CONTAINER=snapshots
SNAPSHOT_BLOB_PREFIX=snapshots
SNAPSHOT_CLEANUP_EVERY_UPLOADS=20

# Snapshot gating
SNAPSHOT_HIGH_CONFIDENCE=0.85
SNAPSHOT_MIN_INTERVAL_SECONDS=20
SNAPSHOT_HIGH_CONFIDENCE_INTERVAL_SECONDS=75
SNAPSHOT_KEEPALIVE_INTERVAL_SECONDS=240
SNAPSHOT_MIN_LUMA=8.0
SNAPSHOT_MIN_FACE_QUALITY=0.08

# Stream
MJPEG_QUALITY=65

# Backend
HOST=0.0.0.0
PORT=8000
CAMERA_INDEX=0

# MongoDB
MONGODB_URI=<your_mongodb_uri>
MONGODB_DB=vyrax_analytics
```

## Setup and Run

### Option A: One-command startup

```powershell
./start-all.ps1
```

### Option B: Manual startup

Backend:

```powershell
cd backend
pip install -r requirements.txt
python web_app.py
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

## Core Runtime Endpoints

- GET /health
	- Service status, camera readiness, MongoDB health.
- GET /emotion
	- Latest inferred emotion payload.
- GET /video_feed
	- Live MJPEG annotated stream.

## Admin Data Endpoints

- GET /api/snapshots?sessionId=<optional>&limit=<optional>
- GET /api/emotions?sessionId=<optional>&limit=<optional>
- GET /api/summary?sessionId=<optional>

These endpoints are the supported integration surface for the admin panel.
Do not connect the frontend directly to MongoDB.

## MongoDB Data Contracts

### snapshots collection

```json
{
	"sessionId": "vyrax-session-xxxx",
	"imageUrl": "https://...",
	"emotion": "happy",
	"confidence": 0.92,
	"timestamp": "2026-04-21T10:20:30.000Z",
	"blobName": "snapshots/2026/04/21/<id>.jpg",
	"contentType": "image/jpeg",
	"sizeBytes": 81234
}
```

### emotions collection

```json
{
	"sessionId": "vyrax-session-xxxx",
	"emotion": "neutral",
	"confidence": 0.73,
	"timestamp": "2026-04-21T10:20:30.000Z"
}
```

### sessions collection

```json
{
	"sessionId": "vyrax-session-xxxx",
	"startTime": "2026-04-21T10:00:00.000Z",
	"endTime": "2026-04-21T10:30:00.000Z",
	"totalEvents": 460,
	"dominantEmotion": "neutral",
	"avgConfidence": 0.78,
	"updatedAt": "2026-04-21T10:30:01.000Z"
}
```

## Judge Demo Script (Recommended)

1. Start system with start-all.ps1.
2. Open live UI and show video + emotion card updates.
3. Change facial expression intentionally to demonstrate transition.
4. Show snapshot events appearing in admin view.
5. Show session summary and explain dominant emotion and average confidence.
6. Explain cloud persistence: image in blob + metadata in MongoDB.

## Safety and Privacy Notes

- This project can process personal visual data; use with informed consent.
- Avoid storing credentials in public repositories.
- Restrict production database and blob access by network and role.

## Troubleshooting

- Camera not opening:
	- Check CAMERA_INDEX and OS camera permissions.
- Slow UI stream:
	- Lower MJPEG_QUALITY.
- Too many snapshots:
	- Increase snapshot interval variables.
- Admin panel showing no data:
	- Verify MongoDB URI, backend health, and /api endpoints.

## Technology Stack

- Frontend: React, Vite, Tailwind CSS, Framer Motion
- Backend: Python, Flask, OpenCV, NumPy
- Storage: Azure Blob Storage, MongoDB

## License

Add your preferred license here before final external publication.
