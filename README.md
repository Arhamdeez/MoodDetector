# Emotion Aware Human Centered Robot Navigation

University project: real-time emotion recognition from webcam using a CNN trained on FER2013.

## Project structure

```
project/
├── dataset/              # Put FER2013 here (see below)
│   └── fer2013.csv
├── models/
│   ├── emotion_cnn.pt    # Saved after training (PyTorch)
│   └── training_curves.png
├── emotion_detection.py  # Webcam emotion detection
├── train_model.py        # Train CNN on FER2013
├── requirements.txt
└── README.md
```

## 1. Environment setup (macOS, VS Code, Python 3)

In VS Code, open the project folder and use the integrated terminal (View → Terminal or `` Ctrl+` ``).

Create and activate a virtual environment (**Python 3.11** recommended):

```bash
cd "/Users/app/Desktop/Semester 8/ML ROBO"
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies (training uses **PyTorch**, not TensorFlow, so you avoid AVX / TensorFlow crashes on some Macs):

```bash
pip install -r requirements.txt
```

If you already have a `.venv`, just activate and install:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Dataset (FER2013 from Kaggle)

1. Download the FER2013 dataset from Kaggle (e.g. [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) or the original competition dataset).
2. Extract the zip. You need a CSV file with columns: **emotion**, **pixels**, **Usage**.
3. Place the CSV as:
   - **`dataset/fer2013.csv`**

If your Kaggle zip contains the file with another name (e.g. `icml_face_data.csv` or `fer2013.csv` in a subfolder), copy or move it to `dataset/fer2013.csv`.

- **emotion**: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral  
- **pixels**: space-separated 48×48 grayscale values (2304 numbers)  
- **Usage**: `Training`, `PublicTest`, or `PrivateTest`

## 3. Train the model

From the project root (with venv activated):

```bash
python train_model.py
```

**Default (more precise):** uses the **full** train/test folders, **data augmentation**, **class weights**, **label smoothing**, up to **40 epochs** with **early stopping**. Expect a longer run on CPU.

**Quick test (small subset, ~10 epochs):**

```bash
python train_model.py --quick
```

**Fine-tune an existing `models/emotion_cnn.pt`:**

```bash
python train_model.py --finetune
```

This will:

- Load images from `dataset/archive/train` and `dataset/archive/test` (or `dataset/fer2013.csv` if no archive)
- Train a small CNN with **PyTorch** and save **`models/emotion_cnn.pt`**
- Save training curves to `models/training_curves.png`

## 4. Run webcam emotion detection

With the model trained:

```bash
python emotion_detection.py
```

- Opens the default webcam (camera 0 on macOS).
- Detects faces with OpenCV’s Haar cascade.
- Runs the CNN on each face and shows **“Emotion: &lt;label&gt;”** (and confidence) on the frame.
- **Press Q** (with the OpenCV window focused) to quit.

If the camera does not open, grant camera access in **System Settings → Privacy & Security → Camera** for Terminal or VS Code.

## 5. Quick reference

| Step              | Command                |
|-------------------|------------------------|
| Activate venv     | `source .venv/bin/activate` |
| Install deps      | `pip install -r requirements.txt` |
| Train model       | `python train_model.py` |
| Run webcam        | `python emotion_detection.py` |
| Quit webcam       | Press **Q**            |

## Dependencies (in requirements.txt)

- **opencv-python** – webcam and face detection  
- **numpy**, **pandas** – data handling  
- **matplotlib** – training curves  
- **torch**, **torchvision** – CNN training and inference (no TensorFlow)  
- **scikit-learn** – optional split if test set is missing  

Compatible with macOS, Python 3, and VS Code.
