# Project Title  
YOLO+ CNN - Hybrid model for Smart parking system.

## 📌 Project Overview  
Goal: Automate parking lot monitoring by providing a reliable real-time solution that helps reduce congestion, save time, and improve traffic management.
This project finds parking slots and classifies them as empty or occupied.

YOLO is used to detect vehicles and parking slot regions.
The detected parking slots are passed into a CNN model, which classifies each slot as empty or occupied.
The hybrid approach improves accuracy compared to using YOLO alone, especially in cases with shadows, partial occlusions, or unclear markings.

## 🗂 Dataset  
- Source: Roboflow 
- Data consists of various images of aprking areas.

## ⚙️ Tech Stack / Tools  
- Python, Open CV, Numpy, Sklearn metrics
- YOLOv8, CNN 

## 🚀 How it Works  
1. Data Collection
  - The dataset is collected from Roboflow – Car Parking Slot Detection.
  - The dataset contains images of parking lots with annotations for cars and parking slots.

2. YOLO Training
  - A YOLOv8 model is trained to detect cars and parking slots.
  - The trained model outputs bounding boxes around vehicles and slot regions in each image/frame.

3. Cropping Detected Regions
  - During testing, YOLO predictions are used to crop the detected parking slot regions.
  - These cropped regions contain either an empty space or a vehicle.

4. CNN Training
  - The cropped slot images are used to train a Convolutional Neural Network (CNN).
  - The CNN learns to classify each slot as: Empty or occupied.

5. For a new image/video:
  - YOLO detects cars and parking slots.
  - Detected slots are cropped and passed to the CNN.
  - The CNN outputs the occupancy status for each slot.

6. Output is visualized as green box for empty slots.

## 📊 Results  
The hybrid model works better than individual YOLO model in identifying and detecting empty parking slots.  

## 👤 Author  
Dia Singh – [LinkedIn](www.linkedin.com/in/dia-singh-31486a220) 

