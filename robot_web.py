"""
Emotion-reactive robot UI: open http://127.0.0.1:8765 in a browser (Chrome recommended).

Uses your webcam in the page; frames are sent to this server for the same PyTorch pipeline
as emotion_detection.py.

Run:  python robot_web.py
"""
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"

app = FastAPI(title="Emotion Robot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor = None


@app.on_event("startup")
def _startup():
    global _predictor
    from emotion_detection import EmotionPredictor

    _predictor = EmotionPredictor()


@app.post("/api/predict")
async def api_predict(image: UploadFile = File(...)):
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty body")
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="could not decode image")
    return _predictor.predict(frame)


@app.get("/")
def index():
    if not (STATIC / "index.html").is_file():
        raise HTTPException(status_code=404, detail="static/index.html missing")
    return FileResponse(STATIC / "index.html")


def main():
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")


if __name__ == "__main__":
    main()
