# Traffic Object Detection Using YOLOv8

AIRI Team PITB | AI Internship Task 1

An end-to-end computer vision project: data collection, annotation review, YOLO-format dataset preparation, training on Google Colab, evaluation, error analysis, inference on images and video, and a small FastAPI demo.

**Author:** <<FILL: your full name>>
**Submitted to:** Omar Farooq, Associate AI/ML Engineer
**Project link (Drive/GitHub):** https://drive.google.com/drive/folders/1OutK2zlnlTy92yoI6arh8x-xc_M7AMEv?usp=sharing

---

## 1. Project Overview

This system detects road vehicles in street-level traffic images and draws a bounding box, class name and confidence score on each one. The final model (`models/best.pt`) works on images and video.

**Why it is useful:** detecting and counting vehicles is the first step of traffic monitoring, congestion analysis and road-safety tools.

**Applications:** roadside cameras, junction monitoring, vehicle counting, smart-city traffic dashboards.

**Classes (5):**

| ID | Class |
|----|-------|
| 0 | bus |
| 1 | car |
| 2 | motorcycle |
| 3 | three-wheeler (rickshaw / CNG) |
| 4 | truck |

Note: the task's Option A lists `person` as a class. The dataset used has no person class and three-wheelers are very common in it, so `three-wheeler` was used as the fifth class instead.

## 2. Dataset

- **Source:** "Road Vehicle Images Dataset" by Ashfak Yeafi (Kaggle), Bangladeshi road vehicle images with YOLO annotations: https://www.kaggle.com/datasets/ashfakyeafi/road-vehicle-images-dataset
- **License:** Open Database License (ODbL) for the database; see the Kaggle page for the contents terms.
- **Size:** 250 images selected from about 2,400 eligible ones (1,559 boxes in total).
- **Class merge:** the 21 source classes were merged into 5 (for example, SUV / minivan / taxi into car; pickup / van into truck; rickshaw / CNG into three-wheeler). Images with other vehicle classes were left out so that no visible vehicle stays unlabeled.
- **Annotation tool:** Roboflow (review and export). 160 of the 250 images were manually reviewed and corrected.
- **Format:** YOLO, one `.txt` per image, `<class_id> <x_center> <y_center> <width> <height>`, all values normalized to 0-1.
- **Split:** 175 train / 50 validation / 25 test (70 / 20 / 10).

## 3. Folder Structure

```
cv_project/
├── dataset/
│   ├── images/{train,val,test}
│   ├── labels/{train,val,test}
│   └── data.yaml
├── notebooks/
│   └── training_notebook.ipynb
├── outputs/
│   ├── training_results/
│   ├── predictions/
│   └── demo_results/
├── models/
│   └── best.pt
├── report/
│   └── final_report.pdf
├── README.md
└── requirements.txt
```

## 4. Training

| Setting | YOLOv8n (final) | YOLOv8s (experiment) |
|---------|-----------------|----------------------|
| Pretrained weights | yolov8n.pt | yolov8s.pt |
| Epochs | 50 | 50 |
| Image size | 640 | 640 |
| Batch size | 8 | 8 |
| Platform / GPU | Google Colab, Tesla T4 | Google Colab, Tesla T4 |
| Training time | 4.0 min | 4.4 min |
| Model size | 6.5 MB | 22.5 MB |

YOLOv8n was chosen as the final model because its accuracy was about the same as YOLOv8s while the file is roughly three times smaller.

## 5. Results

| Model | Split | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-------|-----------|--------|---------|--------------|
| YOLOv8n | val | 0.545 | 0.426 | 0.424 | 0.261 |
| YOLOv8n | test | 0.586 | 0.458 | 0.476 | 0.274 |
| YOLOv8s | val | 0.602 | 0.393 | 0.417 | 0.261 |
| YOLOv8s | test | 0.594 | 0.431 | 0.463 | 0.263 |

- **Best class:** car (mAP@0.5 = 0.513).
- **Weakest classes:** three-wheeler (lowest recall, 0.287) and motorcycle (lowest mAP@0.5:0.95, 0.161).
- The test set is small (25 images), so small differences between models are not reliable.

Confusion matrix and loss curves are in `outputs/training_results/` and in the report.

## 6. Inference

- The final model was run on the 25 unseen test images at confidence 0.35. Results are in `outputs/predictions/test_predictions`.
- Video inference was run on a 28-second street clip (Pexels, https://www.pexels.com/video/32665228/). The output is in `outputs/demo_results/video_demo`.

## 7. Error Analysis (summary)

15 weak or incorrect predictions were reviewed. Main findings:

- Small, distant, hidden or edge-cut vehicles are often missed (false negatives).
- Duplicate boxes appear in crowded scenes (false positives).
- Truck is confused with car and bus; the truck class is too broad because vans and pickups were merged into it.
- Some ground-truth labels overlap or are missing (annotation mistakes).
- Close-up photos, which are rare in the training data, give wrong classes and oversized boxes.

**Suggested improvements:** fix the class merge, add more and harder images (close-up, small, low-light, partly hidden), clean the labels, balance the classes, then try a larger model or more epochs.

## 8. How to Run

### Training (Google Colab)

1. Upload the `cv_project` folder to Google Drive.
2. Open `notebooks/training_notebook.ipynb` in Google Colab and choose a GPU runtime.
3. Mount Drive, install Ultralytics, and run all cells:

```python
from google.colab import drive
drive.mount("/content/drive")
!pip install -U ultralytics
```

### Inference on images

```python
from ultralytics import YOLO

model = YOLO("models/best.pt")
model.predict(source="dataset/images/test", conf=0.35, save=True)
```

### FastAPI demo (local)

```bash
pip install -r requirements.txt
python -m uvicorn main:app --port 8000
```

Open http://127.0.0.1:8000 for the upload page.

- `POST /predict` returns detections as JSON (class, confidence, box).
- `POST /predict-image` returns the image with boxes drawn.

## 9. Tools

Python, Google Colab, Google Drive, YOLOv8, Ultralytics, OpenCV, Roboflow, Matplotlib, Pandas, FastAPI.

## 10. Credits

- Dataset: "Road Vehicle Images Dataset" by Ashfak Yeafi (Kaggle).
- Video: Pexels free stock video (https://www.pexels.com/video/32665228/). Creator: <<FILL: creator name from the Pexels page>>.
