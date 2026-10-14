# Converts the pretrained YOLOv8 Open Images model into the files the web app uses.
# You only need this if you want to re-create the files in the "models" folder;
# the app itself needs no Python.
#
# Setup (once):   pip install ultralytics onnx onnxslim
# Run:            python export_model.py
#
# Nothing is trained here. Ultralytics downloads weights that were already
# trained on Open Images V7 (600 object classes), and we only change the file
# format from PyTorch (.pt) to ONNX, which ONNX Runtime Web can run in a browser.

import json
import os
import shutil

from ultralytics import YOLO

model = YOLO("yolov8s-oiv7.pt")  # "s" = small; downloaded automatically the first time

# imgsz=640: the image size the network expects.
# nms=False: duplicate boxes are removed in script.js instead, because the
#            browser's GPU backend cannot run the built-in version.
onnx_path = model.export(format="onnx", imgsz=640, nms=False, simplify=True, opset=17)

os.makedirs("models", exist_ok=True)
shutil.copy(onnx_path, "models/yolov8s-oiv7.onnx")

# Save the class names in class-number order: labels[0] is class 0, and so on.
labels = [model.names[i] for i in range(len(model.names))]
with open("models/labels.json", "w") as file:
    json.dump(labels, file, indent=0)

print("Saved models/yolov8s-oiv7.onnx and models/labels.json with", len(labels), "classes")
