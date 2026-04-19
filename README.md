# Vyra-X

This repository is organized into two clear modules:

- `frontend/`: React + Tailwind + Framer Motion dashboard UI.
- `backend/`: Python real-time emotion detection pipeline and API.

## Folder Layout

```
vyrax-main/
├─ frontend/
│  ├─ src/
│  └─ package.json
├─ backend/
│  ├─ realtime_emotion.py
│  ├─ emotion_engine.py
│  ├─ config.py
│  ├─ logs/
│  ├─ models/
│  └─ requirements.txt
└─ README.md
```

## Run Frontend

```bash
cd frontend
npm install
npm run dev
```

## Run Backend

```bash
cd backend
pip install -r requirements.txt
python web_app.py
```

## Notes

- Backend paths for logs/reports are now anchored to `backend/`.
- Frontend reads live data from `http://localhost:8000/emotion` and video from `http://localhost:8000/video_feed`.
- Use `./start-all.ps1` from the repo root to launch the backend web server and frontend together in one terminal.
