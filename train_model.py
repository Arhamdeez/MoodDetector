"""
Train a CNN on your FER2013 images (dataset/archive/train|test) using PyTorch.
Works on macOS without TensorFlow/AVX issues.

Run: python train_model.py
Saves: models/emotion_cnn.pt
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
ARCHIVE_DIR = os.path.join(DATASET_DIR, "archive")
TRAIN_DIR = os.path.join(ARCHIVE_DIR, "train")
TEST_DIR = os.path.join(ARCHIVE_DIR, "test")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CSV_PATH = os.path.join(DATASET_DIR, "fer2013.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "emotion_cnn.pt")

EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}
NUM_CLASSES = len(EMOTION_LABELS)
IMG_SIZE = 48
BATCH_SIZE = 64
# Faster training: limit samples per split using a balanced subset.
# (None = use all images in train/test folders)
MAX_TRAIN_SAMPLES = 80 * 64
MAX_TEST_SAMPLES = 20 * 64
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_transforms():
    return transforms.Compose(
        [
            transforms.Grayscale(1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ]
    )


def balanced_subset(ds, max_total, num_classes):
    """
    Take a balanced subset so every emotion folder contributes examples.

    Important: ImageFolder orders samples by class, so naive `range(max_total)`
    would overfit to only the first few classes.
    """
    if max_total is None or len(ds) <= max_total:
        return ds

    per_class = max(1, max_total // num_classes)
    targets = ds.targets  # list[int], class index for each sample in ImageFolder

    selected = []
    for cls_idx in range(num_classes):
        cls_indices = [i for i, t in enumerate(targets) if int(t) == cls_idx]
        selected.extend(cls_indices[:per_class])

    # If still too many (due to rounding), trim deterministically.
    if len(selected) > max_total:
        selected = selected[:max_total]
    return Subset(ds, selected)


def load_from_archive():
    if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(TEST_DIR):
        return None, None
    tfm = get_transforms()
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=tfm)
    test_ds = datasets.ImageFolder(TEST_DIR, transform=tfm)
    train_ds = balanced_subset(train_ds, MAX_TRAIN_SAMPLES, NUM_CLASSES)
    test_ds = balanced_subset(test_ds, MAX_TEST_SAMPLES, NUM_CLASSES)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    return train_loader, test_loader, train_ds


def load_fer2013(csv_path: str):
    path = csv_path
    if not os.path.isfile(path):
        alt = os.path.join(DATASET_DIR, "icml_face_data.csv")
        if os.path.isfile(alt):
            path = alt
        else:
            raise FileNotFoundError(
                f"FER2013 CSV not found. Place fer2013.csv in dataset/."
            )
    df = pd.read_csv(path)
    if "pixels" not in df.columns:
        raise ValueError(f"CSV must have 'pixels'. Found: {list(df.columns)}")
    return df


def preprocess_fer2013(df: pd.DataFrame, usage: str = "Training"):
    usage_col = "Usage" if "Usage" in df.columns else "usage"
    if usage_col not in df.columns:
        cols = df.columns[df.columns.str.lower() == "usage"].tolist()
        usage_col = cols[0] if cols else None
    if usage_col:
        subset = df[df[usage_col].str.strip().str.lower() == usage.lower()]
    else:
        subset = df
    emotions = subset["emotion"].values.astype(np.int64)
    images = []
    for px_str in subset["pixels"]:
        px = np.array(px_str.split(), dtype=np.float32)
        if len(px) != IMG_SIZE * IMG_SIZE:
            px = px[: IMG_SIZE * IMG_SIZE]
        images.append(px.reshape(IMG_SIZE, IMG_SIZE) / 255.0)
    X = np.stack(images)[:, None, :, :]
    return X, emotions


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        n += xb.size(0)
    return total_loss / n, correct / n


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            n += xb.size(0)
    return total_loss / n, correct / n


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Device:", DEVICE)

    train_loader = test_loader = None
    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

    if os.path.isdir(TRAIN_DIR) and os.path.isdir(TEST_DIR):
        print("Training on images in:", TRAIN_DIR, "and validating on:", TEST_DIR)
        if MAX_TRAIN_SAMPLES is not None:
            print(
                f"Subset: up to {MAX_TRAIN_SAMPLES} train / {MAX_TEST_SAMPLES} test samples, {EPOCHS} epochs"
            )
        train_loader, test_loader, train_ref = load_from_archive()
        if isinstance(train_ref, Subset):
            class_to_idx = train_ref.dataset.class_to_idx
        else:
            class_to_idx = train_ref.class_to_idx
        print("Classes:", class_to_idx)
        print("Label map:", EMOTION_LABELS)

        model = EmotionCNN().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        best_val = float("inf")
        best_state = None
        for epoch in range(EPOCHS):
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, criterion, optimizer
            )
            va_loss, va_acc = evaluate(model, test_loader, criterion)
            scheduler.step(va_loss)
            history["loss"].append(tr_loss)
            history["accuracy"].append(tr_acc)
            history["val_loss"].append(va_loss)
            history["val_accuracy"].append(va_acc)
            print(
                f"Epoch {epoch + 1}/{EPOCHS}  train_loss={tr_loss:.4f} acc={tr_acc:.4f}  val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
            )
            if va_loss < best_val:
                best_val = va_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        torch.save(
            {
                "state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "emotion_labels": EMOTION_LABELS,
            },
            MODEL_PATH,
        )
        print("Model saved to", MODEL_PATH)

    else:
        print("Loading CSV:", CSV_PATH)
        df = load_fer2013(CSV_PATH)
        X_train, y_train = preprocess_fer2013(df, usage="Training")
        X_test, y_test = preprocess_fer2013(df, usage="PublicTest")
        if len(X_test) == 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
            )
        X_train_t = torch.from_numpy(X_train.astype(np.float32))
        y_train_t = torch.from_numpy(y_train)
        X_test_t = torch.from_numpy(X_test.astype(np.float32))
        y_test_t = torch.from_numpy(y_test)
        train_loader = DataLoader(
            torch.utils.data.TensorDataset(X_train_t, y_train_t),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        test_loader = DataLoader(
            torch.utils.data.TensorDataset(X_test_t, y_test_t),
            batch_size=BATCH_SIZE,
        )
        model = EmotionCNN().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(EPOCHS):
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, criterion, optimizer
            )
            va_loss, va_acc = evaluate(model, test_loader, criterion)
            history["loss"].append(tr_loss)
            history["accuracy"].append(tr_acc)
            history["val_loss"].append(va_loss)
            history["val_accuracy"].append(va_acc)
            print(
                f"Epoch {epoch + 1}/{EPOCHS}  train_loss={tr_loss:.4f} acc={tr_acc:.4f}  val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
            )

        torch.save(
            {
                "state_dict": model.state_dict(),
                "class_to_idx": {str(i): i for i in range(NUM_CLASSES)},
                "emotion_labels": EMOTION_LABELS,
            },
            MODEL_PATH,
        )
        print("Model saved to", MODEL_PATH)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="Train")
    plt.plot(history["val_accuracy"], label="Val")
    plt.title("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend()
    plot_path = os.path.join(MODELS_DIR, "training_curves.png")
    plt.savefig(plot_path)
    plt.close()
    print("Training curves saved to", plot_path)


if __name__ == "__main__":
    main()
