from ultralytics import YOLO

model = YOLO("runs/classify/train2/weights/best.pt")

results = model(
    r"C:\Users\hp\OneDrive\Desktop\image classification\runs\classify\train2\img.jpg.jpg",
    save=True
)

result = results[0]

print("Predicted Class:", result.names[result.probs.top1])
print("Confidence:", float(result.probs.top1conf) * 100)