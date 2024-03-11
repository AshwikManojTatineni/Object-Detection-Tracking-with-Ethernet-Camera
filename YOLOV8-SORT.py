import numpy as np
from ultralytics import YOLO
import cv2
import math
from sort import *
#from harvesters.core import Harvester


# h = Harvester()
#cap = cv2.VideoCapture("clip.mp4")

# cap = cv2.VideoCapture("rtsp://ASHWIK MANOJ:@169.254.163.9:554/Streaming/Channels/101")
# cap.set(3, 1280)
# cap.set(4, 720)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO('yolov8x.pt')

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3) #max life period which the unmatched_tracker exists, min_hits is the min accepted number which U detect the result (but I do not track).

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

selected_id = None


def on_mouse_click(event, x, y, flags, param):
    global selected_id
    if event == cv2.EVENT_LBUTTONDOWN:
        for result in resultTracker:
            x1, y1, x2, y2, track_id = result
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_id = track_id


cv2.namedWindow("Image")
cv2.setMouseCallback("Image", on_mouse_click)

# Get video information for output
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

while True:
    ret, img = cap.read()
    vid = model(img, stream=True)
    detections = np.empty((0, 5))
    for r in vid:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 3)  # img, 2-co-ordinates, color, thickness
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class name
            cls = int(box.cls[0])
            currentclass = classNames[cls]
            #if currentclass == "person" and conf>0.5:
            cv2.putText(img, f'{currentclass} {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            currentArray = np.array([x1, y1, x2, y2, conf])
            detections = np.vstack((detections, currentArray))

    resultTracker = tracker.update(detections)

    # Highlight the selected box after user clicks
      # Check if resultTracker is not empty
    for result in resultTracker:
        x1, y1, x2, y2, track_id = result
        if track_id == selected_id:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Highlight selected box
            cv2.putText(img, f'{track_id}', (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 3)
            print(track_id)

    cv2.imshow("Image", img)
    out.write(img)  # Write frame to video file

    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break

out.release()  # Release the VideoWriter
cv2.destroyAllWindows()
cap.release()
