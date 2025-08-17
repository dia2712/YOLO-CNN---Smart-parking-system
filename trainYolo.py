from ultralytics import YOLO

# Load a pretrained YOLOv8 model
#model = YOLO("yolov8n.pt")  # Use YOLOv8n for a lightweight model
model=YOLO("yolov8n.pt")
# Train the model

model.train(data=r"C:\Users\diaas\OneDrive\Desktop\capstone2_new\data_new\data.yaml", epochs=15, imgsz=640)