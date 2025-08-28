from ultralytics import YOLO
import cv2

model = YOLO('yolov10x.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    results = model.predict(source=frame, conf=0.25) 

    annotated_frame = results[0].plot()

    cv2.imshow('YOLOv10 Real-Time Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()