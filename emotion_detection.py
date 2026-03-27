"""
Real-time emotion detection from webcam using a CNN trained on your FER2013 images (PyTorch).
Run: python emotion_detection.py
Press Q to quit. Works without a model (face boxes only).
"""
import os
import cv2
import numpy as np
import torch
from collections import deque, Counter

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


def main():
    kind, model, idx_to_emotion = load_emotion_model()
    if kind is None:
        print("No model found. Camera shows face detection only. Run: python train_model.py")
    elif kind == "torch":
        print("PyTorch emotion model loaded.")
    else:
        print("Keras emotion model loaded.")
    if kind == "torch":
        print("Using idx_to_emotion mapping:", idx_to_emotion)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check camera permissions.")
        return

    print("Camera on. Press Q to quit.")
    pred_history = deque(maxlen=8)  # stabilize predictions across frames
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
                # Improve robustness to lighting changes
                face_roi = cv2.equalizeHist(face_roi)
                inp = preprocess_face(face_roi)

                if kind == "torch":
                    with torch.no_grad():
                        t = torch.from_numpy(inp)
                        logits = model(t)
                        probs = torch.softmax(logits, dim=1)[0].numpy()
                    label_idx = int(np.argmax(probs))
                    confidence = float(probs[label_idx])
                else:
                    preds = model.predict(inp, verbose=0)
                    label_idx = int(np.argmax(preds[0]))
                    confidence = float(preds[0][label_idx])

                pred_history.append(label_idx)
                stable_idx = Counter(pred_history).most_common(1)[0][0]
                emotion = idx_to_emotion.get(stable_idx, "Unknown")
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
