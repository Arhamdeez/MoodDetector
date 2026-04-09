"""
Real-time emotion detection from webcam using a CNN trained on your FER2013 images (PyTorch).
Run: python emotion_detection.py
      python emotion_detection.py --camera 1   # if default device fails (common on macOS)
      python emotion_detection.py --temperature 1.5   # if one label (e.g. Disgust) sticks incorrectly
      python emotion_detection.py --sad-bias 0          # disable Sad logit boost
Press Q to quit. Works without a model (face boxes only).
"""
import argparse
import os
import platform
import time

import cv2
import numpy as np
import torch
from collections import deque

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_PT = os.path.join(PROJECT_ROOT, "models", "emotion_cnn.pt")
MODEL_PATH_KERAS = os.path.join(PROJECT_ROOT, "models", "emotion_cnn.keras")

FOLDER_TO_EMOTION = {
    "angry": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",
    "surprise": "Surprise",
}
IMG_SIZE = 48


def _class_index_for_emotion(idx_to_emotion: dict, emotion_name: str):
    """Map display emotion name back to model class index (e.g. Sad -> 5 for folder-trained)."""
    for idx, name in idx_to_emotion.items():
        if name == emotion_name:
            return int(idx)
    return None


def _should_apply_sad_bias_torch(
    logits_1d: torch.Tensor,
    sad_idx: int,
    happy_idx: int | None,
    neutral_idx: int | None,
) -> bool:
    """Only nudge Sad when the model already ranks it plausible.

    Skips closed-mouth / subtle smiles: those usually have Happy competing with Neutral,
    while Sad is not in the top two raw scores.
    """
    L = logits_1d.flatten()
    order = torch.argsort(L, descending=True)
    top2 = set(order[:2].tolist())
    if sad_idx not in top2:
        return False
    if happy_idx is not None and float(L[happy_idx]) >= float(L[sad_idx]):
        return False
    if (
        happy_idx is not None
        and neutral_idx is not None
        and happy_idx in top2
        and float(L[happy_idx]) >= float(L[neutral_idx]) - 0.35
    ):
        return False
    return True


def _should_apply_sad_bias_probs(
    p: np.ndarray,
    sad_idx: int,
    happy_idx: int | None,
    neutral_idx: int | None,
) -> bool:
    """Same gating as torch path, using normalized class probabilities (Keras path)."""
    p = np.asarray(p, dtype=np.float64).ravel()
    p = p / max(float(p.sum()), 1e-12)
    order = np.argsort(-p)
    top2 = set(order[:2].tolist())
    if sad_idx not in top2:
        return False
    if happy_idx is not None and p[happy_idx] >= p[sad_idx]:
        return False
    if (
        happy_idx is not None
        and neutral_idx is not None
        and happy_idx in top2
        and p[happy_idx] >= p[neutral_idx] - 0.05
    ):
        return False
    return True


def _build_torch_model(state_dict):
    from train_model import EmotionCNN

    m = EmotionCNN()
    m.load_state_dict(state_dict)
    m.eval()
    return m


def load_emotion_model():
    """Load PyTorch checkpoint first, then legacy Keras if present.

    Returns: (kind, model, idx_to_emotion)
    """
    if os.path.isfile(MODEL_PATH_PT):
        try:
            ckpt = torch.load(
                MODEL_PATH_PT, map_location="cpu", weights_only=False
            )
        except TypeError:
            ckpt = torch.load(MODEL_PATH_PT, map_location="cpu")
        model = _build_torch_model(ckpt["state_dict"])
        class_to_idx = ckpt.get("class_to_idx", {}) or {}
        idx_to_emotion = {}
        for folder_name, idx in class_to_idx.items():
            idx_to_emotion[int(idx)] = FOLDER_TO_EMOTION.get(
                str(folder_name).lower(), f"Class{idx}"
            )
        return ("torch", model, idx_to_emotion)
    if os.path.isfile(MODEL_PATH_KERAS):
        from tensorflow import keras

        # If using Keras, fall back to a fixed mapping (older runs).
        # Better mapping comes from PyTorch checkpoint (class_to_idx).
        idx_to_emotion = {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Neutral",
            5: "Sad",
            6: "Surprise",
        }
        return ("keras", keras.models.load_model(MODEL_PATH_KERAS), idx_to_emotion)
    return (None, None, {})


def preprocess_face(face_gray: np.ndarray) -> np.ndarray:
    """Resize to 48x48 and normalize to [0,1]. Output shape: (1,1,48,48)."""
    resized = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
    x = resized.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    x = np.expand_dims(x, axis=0)
    return x


def pad_square_crop(gray: np.ndarray, x: int, y: int, w: int, h: int, pad_ratio: float = 0.25):
    """Crop a face ROI as a square with padding, avoiding aspect-ratio distortion."""
    H, W = gray.shape[:2]
    side = max(w, h)
    pad = int(side * pad_ratio)
    cx = x + w // 2
    cy = y + h // 2
    x0 = max(0, cx - side // 2 - pad)
    y0 = max(0, cy - side // 2 - pad)
    x1 = min(W, cx + side // 2 + pad)
    y1 = min(H, cy + side // 2 + pad)
    return gray[y0:y1, x0:x1]


def _try_open_capture(index: int):
    """Open VideoCapture with the best backend for this OS."""
    is_mac = platform.system() == "Darwin"
    if is_mac and hasattr(cv2, "CAP_AVFOUNDATION"):
        return cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(index)


def open_webcam(camera_index: int | None = None):
    """Open the first webcam that returns real frames.

    On macOS, the default backend often fails with plain VideoCapture(0); AVFoundation
    and trying indices 0–2 fixes most cases (Continuity Camera / built-in order).
    """
    if camera_index is not None:
        indices = [camera_index]
    else:
        indices = [0, 1, 2]

    for idx in indices:
        cap = _try_open_capture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        # Warm-up: first reads are often false on macOS until the device is ready.
        ok = False
        for _ in range(30):
            ret, _ = cap.read()
            if ret:
                ok = True
                break
            time.sleep(0.03)
        if ok:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"Webcam OK (device index {idx}).")
            return cap
        cap.release()

    return None


class EmotionPredictor:
    """Single-frame / streaming prediction (used by the web robot UI)."""

    def __init__(
        self,
        temperature: float = 1.0,
        sad_bias: float = 0.18,
        equalize: bool = False,
        prob_history_len: int = 10,
    ):
        self.kind, self.model, self.idx_to_emotion = load_emotion_model()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.sad_idx = _class_index_for_emotion(self.idx_to_emotion, "Sad")
        self.happy_idx = _class_index_for_emotion(self.idx_to_emotion, "Happy")
        self.neutral_idx = _class_index_for_emotion(self.idx_to_emotion, "Neutral")
        self.sb = float(sad_bias)
        self.T = max(0.05, float(temperature))
        self.equalize = equalize
        self.prob_history: deque = deque(maxlen=prob_history_len)
        self.PAD_RATIO = 0.08

    def predict(self, frame_bgr: np.ndarray) -> dict:
        """Run face detect + emotion on one BGR frame (OpenCV)."""
        out: dict = {
            "face_detected": False,
            "emotion": None,
            "confidence": 0.0,
            "probs": {},
            "has_model": self.model is not None,
        }
        if frame_bgr is None or frame_bgr.size == 0:
            return out

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) == 0:
            self.prob_history.clear()
            return out

        out["face_detected"] = True
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])

        if self.model is None:
            return out

        face_roi = pad_square_crop(gray, x, y, w, h, pad_ratio=self.PAD_RATIO)
        if self.equalize:
            face_roi = cv2.equalizeHist(face_roi)
        inp = preprocess_face(face_roi)

        if self.kind == "torch":
            with torch.no_grad():
                t = torch.from_numpy(inp)
                logits = self.model(t)
                if (
                    self.sad_idx is not None
                    and self.sb != 0.0
                    and _should_apply_sad_bias_torch(
                        logits[0], self.sad_idx, self.happy_idx, self.neutral_idx
                    )
                ):
                    logits = logits.clone()
                    logits[0, self.sad_idx] = logits[0, self.sad_idx] + self.sb
                logits = logits / self.T
                probs = torch.softmax(logits, dim=1)[0].numpy()
        else:
            raw = self.model.predict(inp, verbose=0)[0].astype(np.float64)
            raw = np.clip(raw, 1e-8, 1.0)
            raw = raw / raw.sum()
            logp = np.log(raw)
            if self.sad_idx is not None and self.sb != 0.0:
                if _should_apply_sad_bias_probs(
                    raw, self.sad_idx, self.happy_idx, self.neutral_idx
                ):
                    logp = logp.copy()
                    logp[self.sad_idx] = logp[self.sad_idx] + self.sb
            logp = logp / self.T
            logp -= logp.max()
            probs = np.exp(logp)
            probs /= probs.sum()

        self.prob_history.append(probs)
        avg_probs = np.mean(np.stack(self.prob_history, axis=0), axis=0)
        label_idx = int(np.argmax(avg_probs))
        confidence = float(avg_probs[label_idx])
        emotion = self.idx_to_emotion.get(label_idx, "Unknown")

        out["emotion"] = emotion
        out["confidence"] = round(confidence, 4)
        for i, p in enumerate(avg_probs.tolist()):
            name = self.idx_to_emotion.get(i, f"class_{i}")
            out["probs"][name] = round(float(p), 4)
        return out


def main():
    parser = argparse.ArgumentParser(description="Webcam emotion detection")
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        metavar="N",
        help="Camera device index (try 1 if 0 fails)",
    )
    parser.add_argument(
        "--equalize",
        action="store_true",
        help="Apply histogram equalization (often hurts vs training; default is off)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        metavar="T",
        help="Softmax temperature (>1 spreads scores; try 1.3–1.8 if one emotion still dominates)",
    )
    parser.add_argument(
        "--sad-bias",
        type=float,
        default=0.18,
        metavar="B",
        help="Sad logit boost when Sad is already top-2 (0=off; skips subtle smiles vs Neutral)",
    )
    args = parser.parse_args()

    kind, model, idx_to_emotion = load_emotion_model()
    if kind is None:
        print("No model found. Camera shows face detection only. Run: python train_model.py")
    elif kind == "torch":
        print("PyTorch emotion model loaded.")
    else:
        print("Keras emotion model loaded.")
    if kind == "torch":
        print("Using idx_to_emotion mapping:", idx_to_emotion)

    sad_idx = _class_index_for_emotion(idx_to_emotion, "Sad")
    happy_idx = _class_index_for_emotion(idx_to_emotion, "Happy")
    neutral_idx = _class_index_for_emotion(idx_to_emotion, "Neutral")
    sb = float(args.sad_bias)
    if model is not None and sad_idx is not None and sb != 0.0:
        print(
            f"Sad bias: +{sb} when Sad is in top-2 and Happy is not a smile-like rival "
            f"(index {sad_idx}). --sad-bias 0 to disable."
        )
    elif model is not None and sad_idx is None:
        print("Warning: Sad class not found in label map; --sad-bias has no effect.")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = open_webcam(camera_index=args.camera)
    if cap is None:
        print(
            "Could not open any webcam.\n"
            "  • macOS: System Settings → Privacy & Security → Camera → enable for Terminal / Cursor / Python.\n"
            "  • Try another index: python emotion_detection.py --camera 1\n"
            "  • Close other apps using the camera (Zoom, browser, Continuity Camera)."
        )
        return

    print("Camera on. Press Q to quit.")
    # Average softmax vectors (matches train: no histeq by default; temporal mode of argmax biased wrong labels)
    prob_history = deque(maxlen=10)
    T = max(0.05, float(args.temperature))
    PAD_RATIO = 0.08  # less background than before; helps webcam crops

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) > 0:
            # Use the largest face only (reduces jumping between faces)
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            if model is not None:
                face_roi = pad_square_crop(gray, x, y, w, h, pad_ratio=PAD_RATIO)
                # Training uses Resize + ToTensor only; equalizeHist shifts distribution and often biases labels.
                if args.equalize:
                    face_roi = cv2.equalizeHist(face_roi)
                inp = preprocess_face(face_roi)

                if kind == "torch":
                    with torch.no_grad():
                        t = torch.from_numpy(inp)
                        logits = model(t)
                        if (
                            sad_idx is not None
                            and sb != 0.0
                            and _should_apply_sad_bias_torch(
                                logits[0], sad_idx, happy_idx, neutral_idx
                            )
                        ):
                            logits = logits.clone()
                            logits[0, sad_idx] = logits[0, sad_idx] + sb
                        logits = logits / T
                        probs = torch.softmax(logits, dim=1)[0].numpy()
                else:
                    raw = model.predict(inp, verbose=0)[0].astype(np.float64)
                    raw = np.clip(raw, 1e-8, 1.0)
                    raw = raw / raw.sum()
                    logp = np.log(raw)
                    if sad_idx is not None and sb != 0.0:
                        if _should_apply_sad_bias_probs(
                            raw, sad_idx, happy_idx, neutral_idx
                        ):
                            logp = logp.copy()
                            logp[sad_idx] = logp[sad_idx] + sb
                    logp = logp / T
                    logp -= logp.max()
                    probs = np.exp(logp)
                    probs /= probs.sum()

                prob_history.append(probs)
                avg_probs = np.mean(np.stack(prob_history, axis=0), axis=0)
                label_idx = int(np.argmax(avg_probs))
                confidence = float(avg_probs[label_idx])
                emotion = idx_to_emotion.get(label_idx, "Unknown")
                text = f"Emotion: {emotion} ({confidence:.2f})"
            else:
                text = "Face detected (run train_model.py for emotion)"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - th - 12), (x + tw + 4, y), (0, 255, 0), -1)
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
