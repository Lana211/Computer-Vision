# Computer-Vision
# Vehicle Brand Detection, Tracking & Classification

An end-to-end computer vision pipeline built with **Ultralytics YOLO** that:
1. Classifies a car's **brand** (33 classes) from an image using a custom fine-tuned model.
2. Detects and **tracks** vehicles in real video footage.
3. Combines both — for every tracked vehicle in a video, crops it and classifies its brand in real time.
4. Exports the trained model to ONNX for deployment.

This is the capstone project for **Computer Vision for Developers with Ultralytics**.

---

## Project Description

The system identifies vehicle brands in real-world video footage rather than static stock images.
It combines two Ultralytics YOLO tasks:

- **Classification (`yolo11n-cls`)** — fine-tuned on a custom Kaggle dataset of 33 car brands.
- **Detection + Tracking (`yolo11n`, `model.track`)** — detects vehicles in video and assigns each
  a persistent ID across frames using Ultralytics' built-in tracker.

Each tracked vehicle is cropped from the video frame and passed to the classification model, so the
final output video shows every car with a bounding box, track ID, predicted brand, and confidence
score.

## Tasks & Models Used

| Task | Model | Purpose |
|---|---|---|
| Image Classification | `yolo11n-cls.pt` (fine-tuned) | Predict car brand from an image/crop |
| Object Detection + Tracking | `yolo11n.pt` | Detect and track vehicles across video frames |

## Dataset

**[Car Brand Classification Dataset](https://www.kaggle.com/datasets/ahmedelsany/car-brand-classification-dataset)** (Kaggle, Apache-2.0 license)

- 33 car brand classes
- 11,517 training images
- 2,475 validation images
- 2,475 test images

## Pipeline Architecture

```
Video file (own recorded footage)
        │
        ▼
cv2.VideoCapture — read frame by frame
        │
        ▼
yolo11n.pt — model.track() — detect + track vehicles (persistent IDs)
        │
        ▼
Crop each tracked vehicle's bounding box from the frame
        │
        ▼
yolo11n-cls.pt (fine-tuned) — predict brand for each crop
        │
        ▼
Draw box + track ID + predicted brand + confidence on frame
        │
        ▼
cv2.VideoWriter — write annotated frame to output video
```

## Repository Structure

```
.
├── car_brand_capstone.ipynb   # Main notebook: training, evaluation, tracking, export
├── README.md
├── .gitignore
└── results/                   # Output video, confusion matrix, and other evidence files
```

## How to Run

### Prerequisites
- Python 3.10+
- A Kaggle account and API token (for dataset download)
- Google Colab (used for this project) or a local environment with GPU support

### Setup
```bash
pip install ultralytics kaggle
```

### 1. Download the dataset
Set your Kaggle API token as an environment variable (we used Colab Secrets to store
`KAGGLE_API_TOKEN` securely, never hardcoded in the notebook), then:
```bash
kaggle datasets download -d ahmedelsany/car-brand-classification-dataset -p ./data --unzip
```

### 2. Run the notebook
Open `car_brand_capstone.ipynb` and run all cells in order. It will:
- Load and inspect the dataset (train/val/test folders, 33 brand classes)
- Fine-tune `yolo11n-cls.pt` for 5 epochs at `imgsz=224`
- Validate the model and display the confusion matrix
- Run sample predictions (True vs Predicted) on test images
- Run detection + tracking + classification on a vehicle video we recorded ourselves
- Export the trained model to ONNX

### 3. Output produced by this project
- Trained weights: `runs/classify/car_brand_cls/run1/weights/best.pt`
- ONNX export: `runs/classify/car_brand_cls/run1/weights/best.onnx`
- Annotated output video: `output_tracked_final.mp4`
- Confusion matrix: `runs/classify/val/confusion_matrix.png`

## Results

## Deployment

The classification model is exported to **ONNX** format (`best.onnx`) for portable deployment
(e.g. web services, edge devices, or a Streamlit/Gradio front end).

## Training Program Attribution

This project was completed as part of **Computer Vision for Developers with Ultralytics**,
delivered by **SDAIA Academy** via Learning Space (5-day capstone, 30 training hours).

SDAIA Academy on GitHub: https://github.com/SDAIAAcademy

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Car Brand Classification Dataset](https://www.kaggle.com/datasets/ahmedelsany/car-brand-classification-dataset) by Ahmed Elsany (Kaggle)
