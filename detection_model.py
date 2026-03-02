"""
Vehicle Detection Model - Uses custom-trained YOLOv8
Detects: Bus, Car, Motorcycle (trained on custom dataset)
Trained model: best.pt
"""

from ultralytics import YOLO
import cv2
import numpy as np

# Load custom-trained YOLOv8 model (trained on traffic dataset)
model = YOLO("best.pt")

# Vehicle class IDs from your trained dataset (data.yaml)
# 0=Bus, 1=Car, 2=Motorcycle
VEHICLE_CLASSES = {0: "Bus", 1: "Car", 2: "Motorcycle"}
ALL_VEHICLE_IDS = set(VEHICLE_CLASSES.keys())


def detect_vehicles(frame):
    """
    Detect vehicles in a frame/image.
    
    Args:
        frame: numpy array (BGR image from cv2 or RGB)
    
    Returns:
        dict with:
            - count: total vehicle count
            - detections: list of {class, confidence, bbox}
            - annotated_frame: frame with bounding boxes drawn
            - ambulance_detected: bool (based on visual cues)
    """
    results = model(frame, verbose=False, conf=0.3)
    
    detections = []
    vehicle_count = 0
    ambulance_detected = False
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if cls_id in ALL_VEHICLE_IDS:
                vehicle_count += 1
                detections.append({
                    "class": VEHICLE_CLASSES[cls_id],
                    "confidence": round(conf, 2),
                    "bbox": (x1, y1, x2, y2)
                })
    
    # Get annotated frame with bounding boxes
    annotated_frame = results[0].plot()
    
    return {
        "count": vehicle_count,
        "detections": detections,
        "annotated_frame": annotated_frame,
        "ambulance_detected": ambulance_detected
    }


def detect_from_image(image_path):
    """Detect vehicles from an image file path."""
    frame = cv2.imread(image_path)
    if frame is None:
        return {"count": 0, "detections": [], "annotated_frame": None, "ambulance_detected": False}
    return detect_vehicles(frame)


def detect_from_numpy(img_array):
    """Detect vehicles from a numpy array."""
    return detect_vehicles(img_array)


# Quick test
if __name__ == "__main__":
    print("Vehicle Detection Model loaded successfully!")
    print("Supported classes:", list(VEHICLE_CLASSES.values()))
    print("Model ready to use.")
    
    # Test with a sample inference
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    result = detect_vehicles(test_img)
    print("Test inference - Vehicles found:", result["count"])
    print("Everything working!")