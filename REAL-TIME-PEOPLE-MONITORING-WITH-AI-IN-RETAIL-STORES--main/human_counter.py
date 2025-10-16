import cv2
from ultralytics import YOLO
from datetime import datetime

# Load YOLO model
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# Initialize counters
entered = 0
exited = 0
previous_live_count = 0

def get_center(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))
    results = model(frame, verbose=False)
    current_centers = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == 0 and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = get_center(x1, y1, x2, y2)
            current_centers.append((cx, cy))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    live_count = len(current_centers)

    # Detect Entered or Exited
    if live_count > previous_live_count:
        entered += live_count - previous_live_count
    elif live_count < previous_live_count:
        exited += previous_live_count - live_count

    previous_live_count = live_count

    # Display info
    cv2.putText(frame, f"Entered: {entered}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(frame, f"Exited: {exited}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.putText(frame, f"Live Count: {live_count}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("YOLOv8 Room Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

