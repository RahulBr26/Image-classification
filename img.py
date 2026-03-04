from ultralytics import YOLO

model = YOLO('yolov8s-cls.pt')

results = model.train(
    data="custom_dataset",
    epochs=2,
    imgsz=224,
    batch=16,
    device='cpu'  
)

results = model.val()

results = model.predict(
    source="custom_dataset/val",
    save=True
)