# 🤟 Indian Sign Language (ISL) Alphabet Recognizer

A deep learning project for recognizing **Indian Sign Language (ISL) alphabets and digits** using **PyTorch** and **Transfer Learning (ResNet18)**. The project includes a complete machine learning pipeline—from dataset preparation and model training to evaluation and real-time webcam inference.

> **Current Release:** v0.1.0

---

## Features

* Transfer Learning using **ResNet18**
* Automatic train/validation/test split
* Data preprocessing and normalization
* Model checkpointing (best validation accuracy)
* Evaluation on a held-out test set
* Real-time webcam inference using OpenCV
* Hand detection and cropping using MediaPipe
* Modular and extensible project structure

---

## Tech Stack

* Python
* PyTorch
* TorchVision
* OpenCV
* MediaPipe
* Pillow
* NumPy
* Matplotlib

---

## Project Structure

```text
isl-alphabet-recognizer/
│
├── app/
├── data/
│   └── raw/
├── models/
│   └── best_model.pth
├── notebooks/
├── outputs/
├── src/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   ├── transforms.py
│   └── utils.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Dataset

This project uses the **Indian Sign Language (ISLRTC Referred)** dataset available on Kaggle.

Dataset:
https://www.kaggle.com/datasets/atharvadumbre/indian-sign-language-islrtc-referred

> The dataset is **not included** in this repository. Please download it from Kaggle and place it inside:

```text
data/raw/indian-sign-language/
```

The dataset contains **36 classes**:

* Alphabets: **A–Z**
* Digits: **0–9**

---

## Installation

Clone the repository

```bash
git clone https://github.com/PROFFESOR07/isl-alphabet-recognizer.git
cd isl-alphabet-recognizer
```

Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training

Train the model

```bash
python -m src.train
```

The best-performing model is automatically saved to

```text
models/best_model.pth
```

---

## Evaluation

Evaluate the saved model

```bash
python -m src.evaluate
```

Metrics reported:

* Test Loss
* Test Accuracy

---

## Webcam Inference

Launch the webcam recognizer

```bash
python -m src.predict
```

Current implementation:

* Live webcam feed
* MediaPipe hand landmark detection
* Automatic hand cropping
* Real-time prediction
* Confidence score display

---

## Results

Current project status:

* Complete training pipeline
* Validation and testing pipeline
* Model checkpointing
* Real-time webcam prototype
* High accuracy on held-out test samples
* Initial webcam inference implementation

---

## Current Limitations

Although the model performs well on the held-out test set, live webcam predictions are less robust due to differences between the training dataset and real-world webcam images (lighting, background, camera angle, and hand positioning).

This is an active area of improvement.

---

## Future Improvements

* Improved webcam robustness
* Stronger data augmentation
* Fine-tuning additional ResNet layers
* Word-level sign recognition
* Sentence-level recognition
* ONNX/TorchScript export
* Web application deployment
* Mobile deployment

---

## Acknowledgements

* PyTorch
* TorchVision
* OpenCV
* MediaPipe
* Kaggle ISL Dataset by Atharva Dumbre

---

## License

This project is licensed under the **MIT License**.

See the `LICENSE` file for details.
