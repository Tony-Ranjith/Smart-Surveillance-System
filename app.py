import os
import cv2
import numpy as np
import telebot
import time
from flask import Flask, render_template, request, Response,  redirect, url_for
from datetime import datetime, timedelta
from gtts import gTTS
from playsound import playsound
import requests

app = Flask(__name__)

# Telegram bot details

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/object')
def object_detection():
    return render_template('object_detection.html')

@app.route('/crash_detection')  # New route for vehicle crash detection
def vehicle_crash_detection():
    return render_template('crash_detection.html')  # Ensure this HTML file is created in the templates folder

@app.route('/fall_detection')  # New route for fall detection
def fall_detection():
    return render_template('fall_detection.html') 

@app.route('/social_distance')
def social_distance():
    return render_template('social_distance.html')

@app.route('/evidance_detection') #new route for evidance detection
def evidance_detection():
    return render_template('evidance_detection.html')



##CRASH DETECTION

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



##FALL DETECTION......


net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Set a higher confidence threshold to filter detections
CONFIDENCE_THRESHOLD = 0.7

# Directory to save fall frames
SAVE_PATH = "./fall_frames/"

# Ensure the save path exists
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Cooldown mechanism
COOLDOWN_TIME = 60  # 1 minute cooldown after detecting a fall
CAPTURE_FRAME_COUNT = 7  # Capture 5-7 frames after fall detection
person_data = {}  # Dictionary to hold fall detection data for each person

# Thresholds for fall detection
FALL_RATIO_THRESHOLD = 0.6
VERTICAL_FALL_DELTA = 30
HEIGHT_DROP_THRESHOLD = 0.5
HORIZONTAL_DISPLACEMENT_THRESHOLD = 50

# Dictionary to track previous bounding box heights and positions
person_bbox_history = {}

# Telegram bot configuration
TELEGRAM_TOKEN = "bot8042903212:AAEgeKLRypCdXQCha_T8TE6_5dx7swf5vgM"
CHAT_ID = "910245567"

# Global variables for live detection
live_detection_active = False
cap = None


def save_fall_frame_fall(frame, object_id, suffix=''):
    """Save the frame when a fall is detected."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"fall_detected_{object_id}_{timestamp}{suffix}.png"
    cv2.imwrite(os.path.join(SAVE_PATH, filename), frame)
    return os.path.join(SAVE_PATH, filename)


def detect_straight_fall_fall(object_id, current_y_position):
    """Detects falls based on significant downward movement."""
    if object_id in person_bbox_history:
        previous_y = person_bbox_history[object_id]['y']
        delta_y = abs(current_y_position - previous_y)

        if delta_y > VERTICAL_FALL_DELTA:
            return True

    return False


def check_bounding_box_change_fall(object_id, current_bbox_height, current_bbox_x, current_bbox_y):
    """Check for significant changes in bounding box height and horizontal movement."""
    if object_id in person_bbox_history:
        previous_bbox_height = person_bbox_history[object_id]['height']
        previous_bbox_x = person_bbox_history[object_id]['x']

        height_ratio = current_bbox_height / previous_bbox_height if previous_bbox_height != 0 else 1.0
        horizontal_displacement = abs(current_bbox_x - previous_bbox_x)

        if height_ratio < HEIGHT_DROP_THRESHOLD or horizontal_displacement > HORIZONTAL_DISPLACEMENT_THRESHOLD:
            return True

    return False


def send_telegram_notification_fall(saved_frames):
    """Send a notification to Telegram with the detected frames."""
    message = "Fall Detected!\nCaptured Frames:\n Need Help Hurry UP....🏃🏃"
    
    for frame in saved_frames:
        if os.path.exists(frame):
            with open(frame, 'rb') as photo:
                requests.post(f'https://api.telegram.org/{TELEGRAM_TOKEN}/sendPhoto', data={
                    'chat_id': CHAT_ID,
                    'caption': message,
                }, files={'photo': photo})


def cleanup_old_frames_fall(current_time):
    """Delete old frames based on the timestamp."""
    for filename in os.listdir(SAVE_PATH):
        file_path = os.path.join(SAVE_PATH, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > 300:  # 5 minutes
                os.remove(file_path)


def process_frame_fall(frame, detections):
    """Process the frame and detect falls."""
    global person_data, person_bbox_history
    h, w = frame.shape[:2]
    current_time = time.time()

    cleanup_old_frames_fall(current_time)

    saved_frames = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])

        if confidence > CONFIDENCE_THRESHOLD and class_id == 15:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x_start, y_start, x_end, y_end) = box.astype("int")

            bbox_height = y_end - y_start
            bbox_width = x_end - x_start
            bbox_ratio = bbox_height / bbox_width

            if i not in person_bbox_history:
                person_bbox_history[i] = {"height": bbox_height, "x": x_start, "y": y_end}

            if i not in person_data:
                person_data[i] = {"fall_detected": False, "frames_saved": 0, "cooldown_end": 0}

            straight_fall = detect_straight_fall_fall(i, y_end)
            bbox_change = check_bounding_box_change_fall(i, bbox_height, x_start, y_end)

            if bbox_ratio < FALL_RATIO_THRESHOLD or straight_fall or bbox_change:
                if current_time > person_data[i]["cooldown_end"]:
                    if not person_data[i]["fall_detected"]:
                        person_data[i]["fall_detected"] = True
                        person_data[i]["frames_saved"] = 0

                        filename = save_fall_frame_fall(frame, i)
                        saved_frames.append(filename)

                    if person_data[i]["frames_saved"] < CAPTURE_FRAME_COUNT:
                        cv2.putText(frame, "Fall Detected", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)

                        filename = save_fall_frame_fall(frame, i)
                        saved_frames.append(filename)
                        person_data[i]["frames_saved"] += 1

                    if person_data[i]["frames_saved"] >= CAPTURE_FRAME_COUNT:
                        person_data[i]["fall_detected"] = False
                        person_data[i]["cooldown_end"] = current_time + COOLDOWN_TIME
                        send_telegram_notification_fall(saved_frames)

            else:
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

            person_bbox_history[i] = {"height": bbox_height, "x": x_start, "y": y_end}

    return frame


def generate_frames_fall():
    """Generate video frames for the web feed."""
    global cap, live_detection_active
    while live_detection_active:
        ret, frame = cap.read()
        if not ret:
            break

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        output_frame = process_frame_fall(frame, detections)

        ret, buffer = cv2.imencode('.jpg', output_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/start_live_detection', methods=['GET'])
def start_live_detection_fall():
    """Start live detection from the webcam."""
    global live_detection_active, cap
    live_detection_active = True
    cap = cv2.VideoCapture(0)
    return "Live detection started"


@app.route('/stop_live_detection', methods=['GET'])
def stop_live_detection_fall():
    """Stop live detection."""
    global live_detection_active, cap
    live_detection_active = False
    if cap is not None:
        cap.release()
    return "Live detection stopped"


@app.route('/start_video_detection', methods=['POST'])
def start_video_detection_fall():
    """Start video detection from an uploaded file."""
    global cap
    video = request.files['video']
    video_path = os.path.join(SAVE_PATH, video.filename)
    video.save(video_path)

    cap = cv2.VideoCapture(video_path)
    return "Video detection started"


@app.route('/stop_video_detection', methods=['GET'])
def stop_video_detection_fall():
    """Stop video detection."""
    global cap
    if cap is not None:
        cap.release()
    return "Video detection stopped"


@app.route('/video_feed')
def video_feed_fall():
    """Stream the video feed."""
    return Response(generate_frames_fall(), mimetype='multipart/x-mixed-replace; boundary=frame')


##Social Distance......

live_detection_active = False
video_file_path = None

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# YOLO setup for person detection
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')

# Audio alerts dictionary for different languages
alert_messages = {
    'english': 'Social distancing violation detected. Please maintain distance.',
    'hindi': 'सामाजिक दूरी का उल्लंघन हुआ है, कृपया दूरी बनाए रखें।',
    'telugu': 'సామాజిక దూరం ఉల్లంఘించబడింది. దయచేసి దూరం ఉంచండి.',
    'kannada': 'ಸಾಮಾಜಿಕ ಅಂತರ ಉಲ್ಲಂಘಿಸಲಾಗಿದೆ. ದಯವಿಟ್ಟು ಅಂತರವನ್ನು ಉಳಿಸಿಕೊಳ್ಳಿ.'
}

def get_output_layers_social(net):
    layer_names = net.getLayerNames()
    return [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Function to play audio alert in different languages
def play_voice_alert(language):
    if language in alert_messages:
        alert_text = alert_messages[language]
        tts = gTTS(text=alert_text, lang={'english': 'en', 'hindi': 'hi', 'telugu': 'te', 'kannada': 'kn'}[language])
        audio_file = f"alert_{language}.mp3"
        tts.save(audio_file)
        playsound(audio_file)
        os.remove(audio_file)

# Draw bounding boxes with social distancing logic
def draw_bounding_boxes_social(frame, outs, conf_threshold=0.5, nms_threshold=0.4):
    Height, Width = frame.shape[:2]
    class_ids, confidences, boxes = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold and class_id == 0:  # Class ID 0 is 'person'
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Track positions of people and check social distancing
    people_positions = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            people_positions.append((x + w // 2, y + h // 2))
            color = (0, 255, 0)  # Green for compliant people
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Check distances between people
    for i in range(len(people_positions)):
        for j in range(i + 1, len(people_positions)):
            distance = np.linalg.norm(np.array(people_positions[i]) - np.array(people_positions[j]))
            if distance < 100:  # Arbitrary threshold for social distancing
                cv2.line(frame, people_positions[i], people_positions[j], (0, 0, 255), 2)  # Red line for violators
                cv2.putText(frame, "Too Close!", (people_positions[i][0], people_positions[i][1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                for lang in alert_messages.keys():
                    play_voice_alert(lang)

    return frame

# Live feed for social distancing detection
def generate_live_feed_social():
    global live_detection_active
    cap = cv2.VideoCapture(0)  # Webcam capture

    while live_detection_active:
        ret, frame = cap.read()
        if not ret:
            break

        # Detection
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers_social(net))
        frame = draw_bounding_boxes_social(frame, outs)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Video file processing for social distancing detection
def generate_video_feed_social():
    global video_file_path
    cap = cv2.VideoCapture(video_file_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detection
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers_social(net))
        frame = draw_bounding_boxes_social(frame, outs)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Routes

@app.route('/video_feed_social')
def video_feed_social():
    if video_file_path:
        return Response(generate_video_feed_social(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(generate_live_feed_social(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_live_detection_social')
def start_live_detection_social():
    global live_detection_active, video_file_path
    live_detection_active = True
    video_file_path = None  # Ensure live detection is prioritized
    return "Live detection started"

@app.route('/stop_live_detection_social')
def stop_live_detection_social():
    global live_detection_active
    live_detection_active = False
    return "Live detection stopped"

@app.route('/start_video_detection_social', methods=['POST'])
def start_video_detection_social():
    global video_file_path, live_detection_active
    live_detection_active = False  # Stop live detection when a video is uploaded
    video = request.files['video']
    video_file_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(video_file_path)
    return redirect(url_for('index'))

@app.route('/stop_video_detection_social')
def stop_video_detection_social():
    global video_file_path
    video_file_path = None
    return "Video detection stopped"


#evidance detection section

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


###Object Detection

# Telegram bot details
BOT_TOKEN = "8042903212:AAEgeKLRypCdXQCha_T8TE6_5dx7swf5vgM"
CHAT_ID = "910245567"

# Initialize the Telegram bot
bot = telebot.TeleBot(BOT_TOKEN)

# Load YOLO model
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Global variables to track detected objects and movement
currently_detected_objects = {}
max_frames_to_capture = 3  # Maximum frames to capture per detection
object_detected_folder = "object_detected_frames"
if not os.path.exists(object_detected_folder):
    os.makedirs(object_detected_folder)

# Background subtractor to detect movement
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Function to delete frames older than 1 hour
def delete_old_frames(folder, hours=1):
    current_time = datetime.now()
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if current_time - file_mod_time > timedelta(hours=hours):
                os.remove(file_path)
                print(f"Deleted old frame: {file_path}")

# Function to detect objects
def detect_objects(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return indexes, boxes, confidences, class_ids

# Function to send Telegram notification with an image
def send_telegram_photo(photo_path):
    try:
        with open(photo_path, 'rb') as f:
            bot.send_photo(chat_id=CHAT_ID, photo=f, caption="A moving object is detected!")
        print(f"Sent photo to Telegram: {photo_path}")
    except Exception as e:
        print(f"Failed to send photo: {e}")

# Process the frame, detect objects, and handle notifications
def process_frame(frame):
    global currently_detected_objects

    fg_mask = back_sub.apply(frame)
    _, fg_mask_thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
    motion_detected = np.sum(fg_mask_thresh) > 20000

    indexes, boxes, confidences, class_ids = detect_objects(frame)
    detected_this_frame = []

    if len(indexes) > 0 and motion_detected:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            object_id = f"{class_ids[i]}_{x}_{y}_{w}_{h}"

            if object_id not in currently_detected_objects:
                currently_detected_objects[object_id] = {
                    'last_seen': time.time(),
                    'frame_count': 0,
                    'notified': False
                }

            if not currently_detected_objects[object_id]['notified']:
                for _ in range(max_frames_to_capture):
                    frame_name = f"{object_id}_{time.time()}.jpg"
                    frame_path = os.path.join(object_detected_folder, frame_name)
                    cv2.imwrite(frame_path, frame)
                    print(f"Frame saved as: {frame_path}")
                    send_telegram_photo(frame_path)

                currently_detected_objects[object_id]['notified'] = True
                currently_detected_objects[object_id]['last_seen'] = time.time()

            detected_this_frame.append(object_id)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {class_ids[i]}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Remove objects not detected anymore
    for object_id in list(currently_detected_objects.keys()):
        if object_id not in detected_this_frame:
            currently_detected_objects[object_id]['notified'] = False
            currently_detected_objects.pop(object_id)

    return frame

# Resize frame for display
def resize_frame(frame, width=800):
    aspect_ratio = frame.shape[0] / frame.shape[1]
    new_height = int(width * aspect_ratio)
    return cv2.resize(frame, (width, new_height))


# Streaming video feed to the browser
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break

        frame = process_frame(frame)
        frame = resize_frame(frame)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in HTTP multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
    cap.release()

@app.route('/start_live_detection', methods=['GET'])
def start_live_detection():
    print("Live detection started.")
    return "Live detection started."

@app.route('/stop_live_detection', methods=['GET'])
def stop_live_detection():
    print("Live detection stopped.")
    return "Live detection stopped."

@app.route('/start_video_detection', methods=['POST'])
def start_video_detection():
    video = request.files['video']
    video_path = os.path.join("uploads", video.filename)
    video.save(video_path)
    process_video(video_path)  # Implement video processing here
    print("Video detection started.")
    return "Video detection started."

@app.route('/stop_video_detection', methods=['GET'])
def stop_video_detection():
    print("Video detection stopped.")
    return "Video detection stopped."



if __name__ == "__main__":
    app.run(debug=True)

    



