import os
from ultralytics import YOLO
import cv2
import time
import random 
import numpy as np 

# Gerar valores aleatórios para R, G e B entre 0 e 255
r1 = random.randint(0, 255)
g1 = random.randint(0, 255)
b1 = random.randint(0, 255)
r2 = random.randint(0, 255)
g2 = random.randint(0, 255)
b2 = random.randint(0, 255)

video_path = 'gopro360.mp4'
#video_path = 0
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
name = 'ir'

#model = YOLO('openvino\models\yolov8n_openvino_model')

#model_path = ('./mixed_openvino_model')
model_path = ('openvino/yolov8_openvino_model')
# Load a model
model = YOLO(model_path)  # load a custom model
threshold = 0.5
crossing = False
prev_time = 0
class_name_dict = {0: 'hand', 1: 'frunk', 2:'obstacle'}

while ret:
    out.write(frame)
    ret, frame = cap.read()    
    current_time = time.time()
    elapsed_time = current_time - prev_time
    # Verifique se o tempo decorrido é maior que zero para evi tar a divisão por zero
    if elapsed_time > 0:
        fps = 1 / elapsed_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    prev_time = current_time    
    #frame_list.append(frame)
    results = model(frame)[0]
    for result in results.boxes.data.tolist():
            if result[5] == 1:
                x1, y1, x2, y2, score, class_id = result
                y1 = int(y1)
                y2 = int(y2)
                x1 = int(x1)
                x2 = int(x2)
                if score > threshold:
                    cv2.rectangle(frame, ((x1), (y1)), ((x2), (y2)), (r1,g1,b1), 2)
                    cv2.putText(frame, class_name_dict[int(class_id)].upper() + " " + str('%.2f' %(score)), (int(x1), int(y1 - 10)),    
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (r1,g1,b1), 1, cv2.LINE_AA)
            if result[5] == 0:
                xx1, yy1, xx2, yy2, score, class_id = result
                yy1 = int(yy1)
                yy2 = int(yy2)
                xx1 = int(xx1)
                xx2 = int(xx2)
                if score > threshold:
                    cv2.rectangle(frame, ((xx1), (yy1)), ((xx2), (yy2)), (r2,b2,g2), 2)
                    cv2.putText(frame, class_name_dict[int(class_id)].upper() + " " + str('%.2f' %(score)), (int(xx1), int(yy1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (r2,b2,g2), 1, cv2.LINE_AA)
                    #cv2.putText(frame, f"Crossing: {is_crossing}", (10, 30), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
                    if ((y2>yy1>y1) or (y2>yy2>y1)) and ((xx1<x1<xx2) or (xx1<x2<xx2)):
                        cv2.putText(frame, f"CRUZANDO AS LATERAIS", (10, 30), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)

                    elif xx1 > x1 and yy1 >= y1 and xx2 <= x2 and yy2<=y2:
                        cv2.putText(frame, f"Mao totalmente dentro", (10, 30), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
                    elif (yy1<y1<yy2) and ((xx1<x1<xx2)or(x1<xx1<x2) or(xx1<x2<xx2)):
                        cv2.putText(frame, f"Mao na aresta inferior", (10, 30), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)

    cv2.imshow('YOLO', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()