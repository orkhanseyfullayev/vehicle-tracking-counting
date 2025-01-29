import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8l.pt")
font = cv2.FONT_HERSHEY_SIMPLEX
font2 = cv2.FONT_HERSHEY_COMPLEX_SMALL

video_file_name = input("Please enter the video file name (e.g., example.mp4): ")
video_path = f"Videos/{video_file_name}"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Cannot open video file: {video_path}")
    exit()

region_left = np.array([(0, 360), (640, 360), (640, 370), (0, 367)])
region_left = region_left.reshape((-1,1,2))

region_right = np.array([(640,360), (1280, 360), (1280, 350), (640, 350)])
region_right = region_right.reshape(-1,1,2)

incoming_count = set()
outgoing_count = set()

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
        
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.track(rgb_img, persist=True, verbose=False)
    labels = results[0].names

    for i in range(len(results[0].boxes)):
        x1, y1, x2, y2 = results[0].boxes.xyxy[i]
        score = results[0].boxes.conf[i]
        cls = results[0].boxes.cls[i]
        ids = results[0].boxes.id[i]
        x1, y1, x2, y2, score, cls, ids = int(x1), int(y1), int(x2), int(y2), float(score), int(cls), int(ids)
        label = results[0].boxes.cls[i].item()
        name = labels[label]

        if (name == 'car' or name == 'truck') and score >= 0.5:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
            text = f"{name} {score:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), font, 0.5, (255, 255, 255), 1)
        
        name = labels[cls]
        if score < 0.5:
            continue
        
        cx = int(x1/2 + x2/2)
        cy = int(y1/2 + y2/2)

        cv2.line(frame, (0, 360), (1280, 360), (0,0,255), 2)
        
        inside_region_left = cv2.pointPolygonTest(region_left, (cx, cy), False)
        
        if inside_region_left > 0:
            incoming_count.add(ids)
            cv2.circle(frame, (cx, cy), 15, (255, 0, 0), -1)
            cv2.line(frame, (0, 360), (1280, 360), (0,255,255), 3)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            
        inside_region_right = cv2.pointPolygonTest(region_right, (cx, cy), False)
        
        if inside_region_right > 0:
            outgoing_count.add(ids)
            cv2.circle(frame, (cx, cy), 15, (255, 0, 0), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.line(frame, (0, 360), (1280, 360), (0,255,255), 3)
            
    incoming_count_str = 'Left: ' + str(len(incoming_count))
    outgoing_count_str = 'Right: ' + str(len(outgoing_count))
    
    frame[0:80,0:210] = (57, 105, 50)
    if len(incoming_count) >= 10:
        frame[0:80,0:230] = (57, 105, 50)
    frame[0:80,1030:1280] = (57, 105, 50)
    cv2.putText(frame, incoming_count_str, (20, 55), font2, 2, (255,255,255), 1)
    cv2.putText(frame, outgoing_count_str, (1050, 55), font2, 2, (255,255,255), 1)
    if len(outgoing_count) >= 10:
        frame[0:80,1010:1280] = (57, 105, 50)
        cv2.putText(frame, outgoing_count_str, (1030, 55), font2, 2, (255,255,255), 1)
    
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
