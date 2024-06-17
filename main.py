import cv2
import numpy as np
import subprocess
from ultralytics import YOLO
import math

# Initialize the YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush", "gun", "pillow"]

# Define the width and height of the drone's camera resolution
width, height = 2048, 1152
buffer_size = width * height * 3  # 3 bytes per pixel for RGB format

# Start the drone video stream and setup ffmpeg
process = subprocess.Popen(
    ["pylwdrone", "stream", "start", "--out-file", "-"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE  # Capture stderr for debugging
)

ffmpeg_process = subprocess.Popen(
    ["ffmpeg", "-i", "-", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
    stdin=process.stdout,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE  # Capture stderr for debugging
)

try:
    while True:
        # Read the frame from ffmpeg output
        raw_frame = ffmpeg_process.stdout.read(buffer_size)
        if not raw_frame:
            print("No more frames or broken pipe.")
            break

        # Convert the byte data to a numpy array
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))

        # Make frame writable
        frame = frame.copy()

        # Object detection using YOLO
        results = model(frame, stream=True)

        # Process detected objects
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                print("Confidence --->", confidence)

                # Class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                # Annotate frame with class name
                org = (x1, y1 - 10)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)

        # Display the frame
        cv2.imshow('Video Stream', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"Error: {e}")
    # Print stderr from processes for debugging
    print(f"pylwdrone stderr: {process.stderr.read().decode('utf-8')}")
    print(f"ffmpeg stderr: {ffmpeg_process.stderr.read().decode('utf-8')}")
finally:
    # Clean up: terminate the process and close all OpenCV windows
    process.terminate()
    ffmpeg_process.terminate()
    cv2.destroyAllWindows()
