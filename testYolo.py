from ultralytics import YOLO

# Load the trained model
model = YOLO(r"C:\Users\diaas\OneDrive\Desktop\Capstone_code\yolov8n.pt")

# Evaluate the model on validation or test data
metrics = model.val(data=r"C:\Users\diaas\OneDrive\Desktop\Capstone_code\data_new\data.yaml",imgsz=640)
print(metrics)
