import numpy as np
from tensorflow.keras.models import load_model
import cv2

# Load trained CNN model
model = load_model(r"C:\Users\diaas\OneDrive\Desktop\capstone2_new\cnn_model_updated.h5")

# Load test image
image = cv2.imread(r"C:\Users\diaas\OneDrive\Desktop\capstone2_new\matchbox_cars_parkinglot\occupied\roi_0afe47848d83442aab573dcf41543c25_occupied.jpg")

image = cv2.resize(image, (128, 128))  # Resize to match CNN input size
image = image / 255.0  # Normalize (same as training)
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Make Prediction
prediction = model.predict(image)

# Convert prediction to label
label = "Occupied" if prediction[0][0] > 0.5 else "Empty"
print(f"Prediction: {label}")