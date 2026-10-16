import io
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps
from ultralytics import YOLO

app = FastAPI(
    title="Traffic Object Detection API",
    description="YOLOv8n detector for car, motorcycle, bus, truck and three-wheeler. "
                "AIRI Team PITB AI Internship Task 1.",
    version="1.0",
)

# best.pt must be in the same folder as this file
model = YOLO("best.pt")


def read_image(data: bytes):
    """Bytes from the upload -> BGR numpy image (what Ultralytics expects)."""
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "classes": list(model.names.values())}


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Traffic image (jpg/png)"),
    conf: float = Query(0.35, ge=0.05, le=0.95, description="Confidence threshold"),
):
    """Returns the detected objects as JSON (class, confidence, box)."""
    img = read_image(await file.read())
    h, w = img.shape[:2]
    result = model.predict(img, conf=conf, imgsz=640, verbose=False)[0]

    detections, counts = [], {}
    for box in result.boxes:
        name = model.names[int(box.cls[0])]
        x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
        detections.append({
            "class": name,
            "confidence": round(float(box.conf[0]), 3),
            "bbox_xyxy": [x1, y1, x2, y2],
        })
        counts[name] = counts.get(name, 0) + 1

    return {
        "filename": file.filename,
        "image_size": {"width": w, "height": h},
        "confidence_threshold": conf,
        "num_detections": len(detections),
        "counts": counts,
        "detections": detections,
    }


@app.post("/predict-image")
async def predict_image(
    file: UploadFile = File(..., description="Traffic image (jpg/png)"),
    conf: float = Query(0.35, ge=0.05, le=0.95, description="Confidence threshold"),
):
    """Returns the same image with boxes, labels and confidence drawn on it (PNG)."""
    img = read_image(await file.read())
    result = model.predict(img, conf=conf, imgsz=640, verbose=False)[0]
    ok, buf = cv2.imencode(".png", result.plot())
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode the result image.")
    return Response(content=buf.tobytes(), media_type="image/png")
