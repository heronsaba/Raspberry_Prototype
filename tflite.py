import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time


model_path = './models_rasp/model.tflite'

def calculate_fps(start_time, end_time, num_frames):
    elapsed_time = end_time - start_time
    fps = num_frames / elapsed_time
    return fps


BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    max_results=5,
    score_threshold = 0.5,
    running_mode=VisionRunningMode.VIDEO)

detector = vision.ObjectDetector.create_from_options(options)


video_path = 'gopro360.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

num_frames = 0
start_time = time.time()
while cap.isOpened():
    num_frames += 1
    end_time = time.time()
    elapsed_time = end_time - start_time

    if elapsed_time >= 1.0:  # Atualizar o FPS a cada segundo
        fps = calculate_fps(start_time, end_time, num_frames)
        num_frames = 0
        start_time = time.time()
    ret, frame = cap.read()

    if not ret:
        break  # Sai do loop quando todos os frames forem processados ou ocorrer um erro

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)


# Perform object detection on the video frame.
    detection_result = detector.detect_for_video(mp_image, time.time_ns() // 1_000_000).detections

    frunk_detection = next((detection for detection in detection_result if detection.categories[0].category_name == 'frunk'), None)
    hand_detection = next((detection for detection in detection_result if detection.categories[0].category_name == 'hand'), None)

    if frunk_detection:
        frunk_box = frunk_detection.bounding_box
        cv2.rectangle(frame, (frunk_box.origin_x, frunk_box.origin_y), 
                    (frunk_box.origin_x + frunk_box.width, frunk_box.origin_y + frunk_box.height),
                    (0, 255, 0), 2)  # Cor verde, espessura da linha 2
    if hand_detection:
        hand_box = hand_detection.bounding_box
        cv2.rectangle(frame, (hand_box.origin_x, hand_box.origin_y), 
                    (hand_box.origin_x + hand_box.width, hand_box.origin_y + hand_box.height),
                    (0, 0, 255), 2)  # Cor vermelha, espessura da linha 2
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
     # Atualizar o número de frames e calcular FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Detecções', frame)

# Libere os recursos
cap.release()
cv2.destroyAllWindows()