import cv2
from ultralytics import YOLO
import numpy as np
import time 


video_path = r'D:\AI_ML\CarTracking\Y1jTEyb3wiI.mp4'
cap = cv2.VideoCapture(video_path)

model = YOLO('yolov8n.pt')
car_count = 0
previous_centroids_id = {}
car_speeds = {}

line_y = 300
next_id = 1

previous_time = time.time()
PIXEL_TO_METER = 0.05

def euclidean_distance(x1, x2, y1, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

while True:
    isTrue, frame = cap.read()
    if not isTrue:
        break  
    
    current_time = time.time()
    delta_time = current_time - previous_time
    previous_time = current_time
    
    results = model(frame)
    new_centroids = []

    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, confidence, class_id = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            if int(class_id) == 2:  # Class 2 = car
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f'Car: {confidence:.2f}'
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), (0, 0, 255), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                x_centroid = (x1 + x2) // 2
                y_centroid = (y1 + y2) // 2
                new_centroids.append((x_centroid, y_centroid))

    # Assign IDs to centroids
    new_centroids_id = {}
    for (x, y) in new_centroids:
        same_id = None
        min_distance = float('inf')

        for id, (px, py) in previous_centroids_id.items():
            distance = euclidean_distance(x, px, y, py)
            if distance < 50:  # Car threshold
                if distance < min_distance:
                    min_distance = distance
                    same_id = id

        if same_id is not None:
            new_centroids_id[same_id] = (x, y)
        else:
            new_centroids_id[next_id] = (x, y)
            car_speeds[next_id] = 0  # Initialize speed for new car
            next_id += 1

    
    for id, (x, y) in new_centroids_id.items():
        px, py = previous_centroids_id.get(id, (x, y))
        
        distance_meters = euclidean_distance(x, px, y, py) * PIXEL_TO_METER
        speed = distance_meters / delta_time  
        speed_kmh = speed * 3.6 
        car_speeds[id] = speed_kmh
                
        if py < line_y and y >= line_y:  # Car crosses the line
            car_count += 1
        cv2.putText(frame, f'{speed_kmh:.1f} km/h', (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)
    previous_centroids_id = new_centroids_id.copy()

    
   
    cv2.rectangle(frame,(0,0),(40*15,60),(255,255,255),-1)
    cv2.putText(frame, f'Car Count: {car_count}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    frame = cv2.resize(frame, (800, 600))
    cv2.imshow('CarTracking', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()
