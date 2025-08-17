import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load YOLOv8 model for parking slot detection
yolo_model = YOLO(r"C:\Users\diaas\OneDrive\Desktop\Capstone_code\best.pt")

# Load CNN model for classification
cnn_model = load_model(r"C:\Users\diaas\OneDrive\Desktop\Capstone_code\cnn_model_updated.h5")

# Define paths
test_image_folder = r"C:\Users\diaas\OneDrive\Desktop\Capstone_code\data_new\test\images"
test_label_folder = r"C:\Users\diaas\OneDrive\Desktop\Capstone_code\data_new\test\labels"

y_true = []
y_pred = []

# Process each test image
for image_file in sorted(os.listdir(test_image_folder)):
    image_path = os.path.join(test_image_folder, image_file)
    label_path = os.path.join(test_label_folder, image_file.replace(".jpg", ".txt"))

    # Load YOLO labels (Ground Truth)
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()
            class_ids = [int(line.split()[0]) for line in lines]
            true_label = 1 if 1 in class_ids else 0  # If any slot is occupied, mark as 1
            y_true.append(true_label)

    # Read the image
    image = cv2.imread(image_path)

    # Step 1: Use YOLOv8 to Detect Parking Slots
    detections = yolo_model.predict(image_path, conf=0.5)
    
    # Step 2: Extract Bounding Boxes & Classify using CNN
    detected_occupied = False
    for det in detections[0].boxes.xyxy:  # Extract bounding boxes
        x1, y1, x2, y2 = map(int, det)  # Convert to integers
        cropped_slot = image[y1:y2, x1:x2]  # Crop the slot
        
        # Resize & Normalize image for CNN
        cropped_slot = cv2.resize(cropped_slot, (128, 128)) / 255.0
        cropped_slot = np.expand_dims(cropped_slot, axis=0)

        # Classify using CNN
        prediction = cnn_model.predict(cropped_slot)
        if prediction[0][0] > 0.5:
            detected_occupied = True  # Mark image as occupied

    # Final Prediction: If any slot is occupied, mark entire image as 1
    y_pred.append(1 if detected_occupied else 0)

# Step 3: Compute Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

# Print Results
print(f"Hybrid Model Accuracy: {accuracy:.4f}")
print(f"Hybrid Model Precision: {precision:.4f}")
print(f"Hybrid Model Recall: {recall:.4f}")
print(f"Hybrid Model F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

