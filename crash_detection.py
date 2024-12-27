from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import time
import requests

app = Flask(__name__)

# Load YOLOv4 model and class labels
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Telegram configuration
TELEGRAM_TOKEN = '8042903212:AAEgeKLRypCdXQCha_T8TE6_5dx7swf5vgM'
CHAT_ID = '910245567'

# Global variables for control
video_stream = None
detect_live_car = False
detect_video_car = False
video_path_car = None

# Function to send Telegram notification
def send_telegram_notification_car(frames):
    message = "Accident Detected!"
    requests.post(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage', 
                  data={'chat_id': CHAT_ID, 'text': message})

    for frame in frames:
        _, buffer = cv2.imencode('.jpg', frame)
        photo = buffer.tobytes()
        requests.post(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto', 
                      data={'chat_id': CHAT_ID}, files={'photo': photo})

# Function to detect moving objects in a frame
def detect_objects_car(frame):
    height, width = frame.shape[:2]
    resized_frame = cv2.resize(frame, (640, int(640 * height / width)))
    blob = cv2.dnn.blobFromImage(resized_frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.4 and class_id in [2, 3, 5, 7]:  # 2: car, 3: motorcycle, 5: bus, 7: truck
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.6)
    return indexes, boxes, confidences, class_ids

# Function to check overlaps and detect accidents
def check_overlaps_car(boxes, indexes):
    overlap_detected = False
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if i in indexes and j in indexes:
                box1 = boxes[i]
                box2 = boxes[j]
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[0] + box1[2], box2[0] + box2[2])
                y2 = min(box1[1] + box1[3], box2[1] + box2[3])

                if x1 < x2 and y1 < y2:
                    overlap_detected = True
                    break

    return overlap_detected

# Function to draw labels
def draw_labels_car(indexes, boxes, class_ids, confidences, frame):
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {round(confidence, 2)}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Video generator for live or uploaded video
def gen_frames_car(live=True, path=None):  
    global detect_live_car, detect_video_car
    if live:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(path)

    while cap.isOpened():
        if not (detect_live_car or detect_video_car):  # If detection is stopped, break out of loop
            break

        ret, frame = cap.read()
        if not ret:
            break

        indexes, boxes, confidences, class_ids = detect_objects_car(frame)
        accident_detected = check_overlaps_car(boxes, indexes)

        if accident_detected:
            send_telegram_notification_car([frame])

        draw_labels_car(indexes, boxes, class_ids, confidences, frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index_car():
    return render_template('crash_detection.html')

@app.route('/video_feed_car')
def video_feed_car():
    global detect_live_car, detect_video_car, video_path_car
    if detect_live_car:
        return Response(gen_frames_car(live=True), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif detect_video_car:
        return Response(gen_frames_car(live=False, path=video_path_car), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_live_detection_car')
def start_live_detection_car():
    global detect_live_car, detect_video_car
    detect_live_car = True
    detect_video_car = False
    return "Live detection started"

@app.route('/stop_live_detection_car')
def stop_live_detection_car():
    global detect_live_car
    detect_live_car = False
    return "Live detection stopped"

@app.route('/start_video_detection_car', methods=['POST'])
def start_video_detection_car():
    global detect_live_car, detect_video_car, video_path_car
    if 'video' in request.files:
        video = request.files['video']
        video_path_car = 'uploaded_video.mp4'
        video.save(video_path_car)
        detect_live_car = False
        detect_video_car = True
        return "Video detection started"

@app.route('/stop_video_detection_car')
def stop_video_detection_car():
    global detect_video_car
    detect_video_car = False
    return "Video detection stopped"

if __name__ == '__main__':
    app.run(debug=True)
