from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = "/Users/nikita/Documents/GitHub/hacaton1/yolo11x.pt"
DATA_CONFIG = Path("/Users/nikita/Documents/GitHub/hacaton1/yolo_learning/person_learning/data.yaml")

model = YOLO(MODEL_PATH)

model.train(
    data=str(DATA_CONFIG),
    epochs = 100,
    imgsz = 640,
    batch = 4,
    device = "mps",
    optimizer= "auto",
)
