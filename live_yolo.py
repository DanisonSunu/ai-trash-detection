from picamera2 import Picamera2
from ultralytics import YOLO
import cv2

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (640, 480)}))
picam2.start()

# Load YOLO model (you can change 'yolov8n.pt' to your trained model if any)
model = YOLO("yolov8n.pt")

print("Starting live detection... Press 'q' to quit.")

while True:
    # Capture frame
    frame = picam2.capture_array()

    # Run YOLO inference
    results = model(frame, stream=True)

    # Draw results on the frame
    for r in results:
        annotated_frame = r.plot()

        # Show frame
        cv2.imshow("YOLO Live Detection", annotated_frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
picam2.stop()
cv2.destroyAllWindows()
