from ultralytics import YOLO
import cv2
import os


# load the YOLOv8 model
model = YOLO("yolov8n.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "boat", "traffic light",
               "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
               "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
               "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
               "tennis racket8", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
               "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
               "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote",
               "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
               "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# Load the Video
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the video file relative to the script's directory
video_filename = 'lane_detected.mp4' 
video_path = os.path.join(script_dir, video_filename)


cap = cv2.VideoCapture(video_path)

# Read Frames from the Video
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (900,500))
    # Detect and track Objects
    results = model(frame, verbose=True, stream=True, conf=0.4)

    # plot the Result
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cls = int(box.cls[0])
            cv2.putText(frame, classNames[cls], [x1, y1], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # Visualize
    cv2.imshow('output', frame)

    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
