import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Load YOLO model for detecting parking slots and cars
yolo_model = YOLO(r"C:\Users\diaas\OneDrive\Desktop\Capstone_code\best.pt")

# Load CNN model for classifying parking slots as empty or occupied
cnn_model = load_model(r"C:\Users\diaas\OneDrive\Desktop\Capstone_code\cnn_model_updated.h5")

# Load video file
cap = cv2.VideoCapture(r"C:\Users\diaas\OneDrive\Desktop\Capstone_code\InputVideo.mp4")

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Error: Could not read frame from camera.")
        break  

    # Run YOLO inference
    results = yolo_model(frame)

    # Extract bounding boxes from YOLO
    detections = results[0].boxes.data.cpu().numpy()  # Get bounding box predictions

    for detection in detections:
        x1, y1, x2, y2, conf, class_id = map(int, detection[:6])  # Extract values
        
        # Ensure we're detecting parking slots (Adjust class_id based on YOLO training)
        if class_id != 1:  # Modify class_id if necessary
            continue

        # Crop detected slot
        cropped_slot = frame[y1:y2, x1:x2]

        # Skip if invalid crop
        if cropped_slot.size == 0:
            continue  

        # Preprocess for CNN
        slot_img = cv2.resize(cropped_slot, (128, 128)) / 255.0
        slot_img = np.expand_dims(slot_img, axis=0)

        # Predict Empty/Occupied
        prediction = cnn_model.predict(slot_img)[0][0]

        # Set default color
        color = (255, 255, 255)  # White (default in case of unexpected prediction)

        # Determine label and color
        if prediction > 0.5:
            label = "Empty"
            color = (0, 255, 0)  # Green for empty slots
        else:
            label = "Occupied"
            color = (0, 0, 255)  # Red for occupied slots

        # Draw bounding box & label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Write frame to output video
    output_video.write(frame)

    # Show output in real-time
    cv2.imshow("Live Parking Slot Detection", frame)

    # Press 'q' to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

# Release resources
cap.release()
output_video.release()
cv2.destroyAllWindows()

