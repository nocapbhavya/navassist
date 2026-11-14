"""Re-downloads the YOLOv8n and MiDaS ONNX models into models/ if they are missing.
The three PP-OCRv4 models (ppocrv4_det.onnx, ppocr_cls.onnx, ppocrv4_rec.onnx) ship with the project."""
import os, urllib.request

MODELS = {
    # COCO-pretrained YOLOv8n exported to ONNX (input 1x3x640x640, output 1x84x8400).
    # Alternative: pip install ultralytics && yolo export model=yolov8n.pt format=onnx
    "yolov8n.onnx": "https://raw.githubusercontent.com/Hyuto/yolov8-onnxruntime-web/master/public/model/yolov8n.onnx",
    # MiDaS v2.1 small, official release from Intel ISL
    "model-small.onnx": "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx",
}

os.makedirs("models", exist_ok=True)
for name, url in MODELS.items():
    path = os.path.join("models", name)
    if os.path.exists(path):
        print("exists:", path); continue
    print("downloading", name, "...")
    urllib.request.urlretrieve(url, path)
print("done")
