# Emotion & Attention Analyzer

Real-time computer vision system that detects faces, recognizes emotions and evaluates attention levels through head pose estimation — built with Python and Deep Learning.

---

## Tech Stack

`Python` · `OpenCV` · `MediaPipe` · `TensorFlow` · `NumPy` · `Docker`

---

## What it does

- Detects faces in real-time video streams or static images
- Classifies basic emotions using a pre-trained deep learning model
- Evaluates attention level via head pose and gaze estimation
- Modular architecture — each component runs independently

---

## Project Structure

```
├── main.py          # Entrypoint
├── detector.py      # Face detection module
├── emotion.py       # Emotion recognition module
├── headpose.py      # Head pose / attention estimation
├── tracker.py       # Object tracking
├── utils.py         # Shared utilities
└── requirements.txt
```

---

## Quickstart

```bash
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate        # Windows
source venv/bin/activate       # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python main.py
```

> Requires a webcam for real-time mode.

---

## Technical Decisions

- **Face detection:** OpenCV + MediaPipe for real-time performance on standard hardware
- **Emotion recognition:** pre-trained deep learning model fine-tuned on facial expression datasets
- **Attention estimation:** head pose angles (pitch, yaw, roll) via MediaPipe landmarks as attention proxy
- **Modular design:** each module (detector, emotion, headpose) is decoupled for easy replacement or upgrade with state-of-the-art models
- **Edge-ready architecture:** lightweight pipeline designed to run on constrained hardware
