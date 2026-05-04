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
# Built before uvicorn starts (see main()). Web defaults favor fewer false "Fear" hits:
# no expressive_bias (that nudges Angry/Fear when #2), small sad_bias when Sad is close #2.
_predictor_kwargs: dict = {
    "temperature": 1.1,
    "sad_bias": 0.08,
    "equalize": False,
    "prob_history_len": 5,
    "expressive_bias": 0.0,
    "force_cpu": False,
}


@app.on_event("startup")
def _startup():
    global _predictor
    from emotion_detection import EmotionPredictor

    _predictor = EmotionPredictor(**_predictor_kwargs)


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
    import argparse
    import uvicorn

    global _predictor_kwargs

    p = argparse.ArgumentParser(
        description="Web UI for emotion robot. Same model as emotion_detection.py; "
        "defaults here disable expressive_bias (less Fear inflation) and apply a small sad_bias."
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--temperature",
        type=float,
        default=1.1,
        metavar="T",
        help="Softmax temperature (default 1.1)",
    )
    p.add_argument(
        "--smooth-frames",
        type=int,
        default=5,
        metavar="N",
        dest="smooth_frames",
        help="Average probs over last N frames (default 5)",
    )
    p.add_argument(
        "--expressive-bias",
        type=float,
        default=0.0,
        metavar="B",
        dest="expressive_bias",
        help="Nudge Angry/Fear when close #2 (default 0 for web; try 0.07 to match CLI demo)",
    )
    p.add_argument(
        "--sad-bias",
        type=float,
        default=0.08,
        metavar="B",
        dest="sad_bias",
        help="Boost Sad when it is close #2 (default 0.08; set 0 to disable)",
    )
    p.add_argument(
        "--equalize",
        action="store_true",
        help="Histogram equalization on face ROI (often hurts vs training)",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference",
    )
    args = p.parse_args()

    _predictor_kwargs = {
        "temperature": args.temperature,
        "sad_bias": args.sad_bias,
        "equalize": args.equalize,
        "prob_history_len": max(1, args.smooth_frames),
        "expressive_bias": args.expressive_bias,
        "force_cpu": args.cpu,
    }

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
