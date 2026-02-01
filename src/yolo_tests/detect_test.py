import cv2
from ultralytics import YOLO
import os

# Correct absolute path to test.jpg
img_path = os.path.join("data", "test.jpg")

# Load image
img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(f"Image not found at: {img_path}")

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Run detection
results = model(img)

# Plot detections on the original image
annotated_img = results[0].plot()

# ---- RESIZE IMAGE TO FIT SCREEN ----
screen_width, screen_height = 1280, 720
scale_w = screen_width / annotated_img.shape[1]
scale_h = screen_height / annotated_img.shape[0]
scale = min(scale_w, scale_h)

new_w = int(annotated_img.shape[1] * scale)
new_h = int(annotated_img.shape[0] * scale)
display_img = cv2.resize(annotated_img, (new_w, new_h))
# -------------------------------------

# Show result
cv2.imshow("YOLO Detection Test", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
