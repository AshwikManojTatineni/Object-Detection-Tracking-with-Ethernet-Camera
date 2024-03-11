import numpy as np
import cv2
import math
from tracker1 import Tracker
from hik_camera.hik_camera.hik_camera import HikCamera
from ultralytics import YOLO

ips = HikCamera.get_all_ips()
ip = ips[0]

cam = HikCamera(ip)

model = YOLO('yolov8x.pt')

tracker = Tracker()
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
        print("Mouse Click Event")
        for track in tracker.tracks:
            if track.check_inside(x, y):
                selected_id = track.track_id
                print(f'Selected Track ID: {selected_id}')

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", on_mouse_click)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

with cam:
    cam["ExposureAuto"] = "Off"
    cam["ExposureTime"] = 20000

    while True:
        bgr = cam.robust_get_frame()
        rgb_in = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.flip(rgb_in, -1)
        vid = model(rgb, stream=True)
        detections = np.empty((0, 5))
        for r in vid:
            boxes = r.boxes
            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 0, 0), 3)  # img, 2-co-ordinates, color, thickness
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class name
                cls = int(box.cls[0])
                currentclass = classNames[cls]
                if currentclass == "person" and conf > 0.5:
                    cv2.putText(rgb, f'{currentclass} {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                                2)
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        tracker.update(rgb, detections)
        result_tracker = tracker
        # Highlight the selected box after user clicks
        for track in tracker.tracks:
            x1, y1, x2, y2 = track.bbox  # Get bounding box coordinates
            track_id = track.track_id  # Get track ID
            if track_id == selected_id:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Highlight selected box
                cv2.putText(rgb, f'{track_id}', (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 3)
                print(f'Selected Track ID: {track_id}')

        cv2.imshow("Image", rgb)
        out.write(rgb)  # Write frame to video file

        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' to exit
            break

    out.release()  # Release the VideoWriter
    cv2.destroyAllWindows()


