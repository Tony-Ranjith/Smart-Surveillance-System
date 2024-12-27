from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import requests

app = Flask(__name__)

# Load YOLOv4 model for evidence detection
model_config = "yolov4.cfg"  # Path to your YOLO configuration file
model_weights = "yolov4.weights"  # Path to your YOLO weights file
net = cv2.dnn.readNet(model_weights, model_config)

# Define the classes for evidence detection
evidence_classes = ['weapon', 'knife', 'gun', 'explosives', 'ammunition', 'drug paraphernalia', 'fire']
CONFIDENCE_THRESHOLD = 0.5

# Get the output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Telegram Bot Details
TELEGRAM_TOKEN = 'bot8042903212:AAEgeKLRypCdXQCha_T8TE6_5dx7swf5vgM'
CHAT_ID = '910245567'

# Global variables for control
video_stream = None
detect_live_evidence = False
detect_video_evidence = False
video_path_evidence = None

# Function to send Telegram notification with frame image
def send_telegram_alert(image):
    url = f'https://api.telegram.org/{TELEGRAM_TOKEN}/sendPhoto'
    _, img_encoded = cv2.imencode('.jpg', image)
    files = {'photo': ('image.jpg', img_encoded.tobytes())}
    data = {'chat_id': CHAT_ID, 'caption': 'Evidence Detected'}
    
    # Debugging: Log request details
    print("Sending notification...")

    try:
        response = requests.post(url, data=data, files=files)
        if response.status_code == 200:
            print("Notification sent successfully.")
        else:
            print(f"Failed to send notification: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Error while sending notification: {e}")

    return response.status_code

def detect_objects(frame, frame_id):
    original_height, original_width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONFIDENCE_THRESHOLD and class_id < len(evidence_classes):
                center_x = int(detection[0] * original_width)
                center_y = int(detection[1] * original_height)
                w = int(detection[2] * original_width)
                h = int(detection[3] * original_height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            box = boxes[i]
            x, y, w, h = box
            label = str(evidence_classes[class_ids[i]])
            color = (0, 255, 0)

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Send a Telegram alert with the detected frame image (every 30 frames to reduce spam)
            if frame_id % 30 == 0:
                print(f"Detected {label}. Sending notification...")
                send_telegram_alert(frame)

    return frame

# Video generator for live or uploaded video
def gen_frames_evidence(live=True, path=None):  
    global detect_live_evidence, detect_video_evidence
    if live:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(path)

    frame_id = 0
    while cap.isOpened():
        if not (detect_live_evidence or detect_video_evidence):  # If detection is stopped, break out of loop
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        # Perform object detection
        frame = detect_objects(frame, frame_id)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index_evidence():
    return render_template('evidance_detection.html')

@app.route('/video_feed_evidence')
def video_feed_evidence():
    global detect_live_evidence, detect_video_evidence, video_path_evidence
    if detect_live_evidence:
        return Response(gen_frames_evidence(live=True), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif detect_video_evidence:
        return Response(gen_frames_evidence(live=False, path=video_path_evidence), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_live_detection_evidence')
def start_live_detection_evidence():
    global detect_live_evidence, detect_video_evidence
    detect_live_evidence = True
    detect_video_evidence = False
    return "Live detection started"

@app.route('/stop_live_detection_evidence')
def stop_live_detection_evidence():
    global detect_live_evidence
    detect_live_evidence = False
    return "Live detection stopped"

@app.route('/start_video_detection_evidence', methods=['POST'])
def start_video_detection_evidence():
    global detect_live_evidence, detect_video_evidence, video_path_evidence
    if 'video' in request.files:
        video = request.files['video']
        video_path_evidence = 'uploaded_video.mp4'
        video.save(video_path_evidence)
        detect_live_evidence = False
        detect_video_evidence = True
        return "Video detection started"

@app.route('/stop_video_detection_evidence')
def stop_video_detection_evidence():
    global detect_video_evidence
    detect_video_evidence = False
    return "Video detection stopped"

if __name__ == '__main__':
    app.run(debug=True)
















