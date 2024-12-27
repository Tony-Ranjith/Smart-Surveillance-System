from flask import Flask, Response, request, render_template
import cv2
import numpy as np
import time
import os
import requests

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained MobileNet SSD model from OpenCV
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


@app.route('/')
def index_fall():
    """Render the main page."""
    return render_template('fall_detection.html')


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


if __name__ == '__main__':
    app.run(debug=True)







