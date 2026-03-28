"""
Train a CNN on FER2013 images (dataset/archive/train|test) using PyTorch.

Default: full dataset + augmentation + class weights + early stopping (more precise).
Quick:    python train_model.py --quick
Fine-tune existing model: python train_model.py --finetune

Saves: models/emotion_cnn.pt
"""
import argparse
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

# Quick mode (small balanced subset)
QUICK_TRAIN_MAX = 80 * 64
QUICK_TEST_MAX = 20 * 64
QUICK_EPOCHS = 10

# Precision mode (default)
FULL_EPOCHS = 40
EARLY_STOP_PATIENCE = 8
FINETUNE_EPOCHS = 25
FINETUNE_LR = 3e-4
BASE_LR = 1e-3
LABEL_SMOOTHING = 0.08


def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon: use Metal (GPU) so CPU stays cooler than pure CPU training
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _pick_device()


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


def get_val_transforms():
    return transforms.Compose(
        [
            transforms.Grayscale(1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ]
    )


def get_train_transforms():
    """Augmentation helps webcam: lighting, pose, mirroring."""
    return transforms.Compose(
        [
            transforms.Grayscale(1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.RandomAffine(
                degrees=0, translate=(0.07, 0.07), scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
        ]
    )


def balanced_subset(ds, max_total, num_classes):
    if max_total is None or len(ds) <= max_total:
        return ds
    per_class = max(1, max_total // num_classes)
    targets = ds.targets
    selected = []
    for cls_idx in range(num_classes):
        cls_indices = [i for i, t in enumerate(targets) if int(t) == cls_idx]
        selected.extend(cls_indices[:per_class])
    if len(selected) > max_total:
        selected = selected[:max_total]
    return Subset(ds, selected)


def class_weights_from_folder(train_dir):
    """Inverse-frequency weights (helps rare classes like disgust)."""
    tmp = datasets.ImageFolder(train_dir, transform=get_val_transforms())
    counts = torch.bincount(
        torch.tensor(tmp.targets, dtype=torch.long), minlength=NUM_CLASSES
    ).float()
    counts = counts.clamp(min=1.0)
    w = counts.sum() / (NUM_CLASSES * counts)
    w = w / w.mean()
    return w.to(DEVICE)


def build_archive_loaders(quick: bool):
    tfm_train = get_train_transforms() if not quick else get_val_transforms()
    train_full = datasets.ImageFolder(TRAIN_DIR, transform=tfm_train)
    test_full = datasets.ImageFolder(TEST_DIR, transform=get_val_transforms())
    class_to_idx = train_full.class_to_idx

    if quick:
        train_ds = balanced_subset(train_full, QUICK_TRAIN_MAX, NUM_CLASSES)
        test_ds = balanced_subset(test_full, QUICK_TEST_MAX, NUM_CLASSES)
    else:
        train_ds, test_ds = train_full, test_full

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    return train_loader, test_loader, class_to_idx


def load_fer2013(csv_path: str):
    path = csv_path
    if not os.path.isfile(path):
        alt = os.path.join(DATASET_DIR, "icml_face_data.csv")
        if os.path.isfile(alt):
            path = alt
        else:
            raise FileNotFoundError(
                "FER2013 CSV not found. Place fer2013.csv in dataset/."
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


def train_loop(
    model,
    train_loader,
    test_loader,
    class_to_idx,
    epochs,
    lr,
    use_class_weights,
    use_label_smoothing,
    early_stop_patience,
    class_weight_override=None,
):
    if class_weight_override is not None:
        cw = class_weight_override.to(DEVICE)
        print("Class weights (from labels):", cw.cpu().numpy().round(3))
    elif use_class_weights and os.path.isdir(TRAIN_DIR):
        cw = class_weights_from_folder(TRAIN_DIR)
        print("Class weights:", cw.cpu().numpy().round(3))
    else:
        cw = None

    try:
        criterion = nn.CrossEntropyLoss(
            weight=cw,
            label_smoothing=LABEL_SMOOTHING if use_label_smoothing else 0.0,
        )
    except TypeError:
        criterion = nn.CrossEntropyLoss(weight=cw)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6
    )

    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = evaluate(model, test_loader, criterion)
        scheduler.step(va_loss)
        history["loss"].append(tr_loss)
        history["accuracy"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_accuracy"].append(va_acc)
        print(
            f"Epoch {epoch + 1}/{epochs}  train_loss={tr_loss:.4f} acc={tr_acc:.4f}  "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
        )
        if va_loss < best_val - 1e-5:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if early_stop_patience and no_improve >= early_stop_patience:
            print(f"Early stopping (no val improvement for {early_stop_patience} epochs).")
            break

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
    return history


def main():
    parser = argparse.ArgumentParser(description="Train emotion CNN (PyTorch)")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small balanced subset + 10 epochs (fast debug)",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Load models/emotion_cnn.pt and train further with lower LR",
    )
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Device:", DEVICE)

    if os.path.isdir(TRAIN_DIR) and os.path.isdir(TEST_DIR):
        quick = args.quick
        print("Training on:", TRAIN_DIR)
        print("Validation on:", TEST_DIR)
        if quick:
            print(
                f"Mode: QUICK — up to {QUICK_TRAIN_MAX} train / {QUICK_TEST_MAX} test, {QUICK_EPOCHS} epochs"
            )
        else:
            print(
                f"Mode: PRECISION — full train/test, augmentation, class weights, "
                f"up to {FULL_EPOCHS} epochs, early stop patience {EARLY_STOP_PATIENCE}"
            )
        if args.finetune:
            print("Fine-tune: will load existing checkpoint if present.")

        train_loader, test_loader, class_to_idx = build_archive_loaders(quick=quick)
        print("Classes:", class_to_idx)

        model = EmotionCNN().to(DEVICE)
        if args.finetune and os.path.isfile(MODEL_PATH):
            try:
                ckpt = torch.load(
                    MODEL_PATH, map_location=DEVICE, weights_only=False
                )
            except TypeError:
                ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(ckpt["state_dict"])
            print("Loaded checkpoint for fine-tuning.")

        if quick:
            history = train_loop(
                model,
                train_loader,
                test_loader,
                class_to_idx,
                epochs=QUICK_EPOCHS,
                lr=BASE_LR,
                use_class_weights=False,
                use_label_smoothing=False,
                early_stop_patience=0,
            )
        elif args.finetune and os.path.isfile(MODEL_PATH):
            history = train_loop(
                model,
                train_loader,
                test_loader,
                class_to_idx,
                epochs=FINETUNE_EPOCHS,
                lr=FINETUNE_LR,
                use_class_weights=True,
                use_label_smoothing=True,
                early_stop_patience=6,
            )
        else:
            history = train_loop(
                model,
                train_loader,
                test_loader,
                class_to_idx,
                epochs=FULL_EPOCHS,
                lr=BASE_LR,
                use_class_weights=True,
                use_label_smoothing=True,
                early_stop_patience=EARLY_STOP_PATIENCE,
            )

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
        counts = torch.bincount(y_train_t, minlength=NUM_CLASSES).float().clamp(min=1.0)
        cw_csv = (counts.sum() / (NUM_CLASSES * counts)).to(DEVICE)
        cw_csv = cw_csv / cw_csv.mean()
        history = train_loop(
            model,
            train_loader,
            test_loader,
            {str(i): i for i in range(NUM_CLASSES)},
            epochs=QUICK_EPOCHS if args.quick else FULL_EPOCHS,
            lr=BASE_LR,
            use_class_weights=True,
            use_label_smoothing=True,
            early_stop_patience=EARLY_STOP_PATIENCE if not args.quick else 0,
            class_weight_override=cw_csv,
        )

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
