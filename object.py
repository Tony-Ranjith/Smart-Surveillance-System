from flask import Flask, Response, render_template, request
import cv2
import numpy as np
import telebot
import time
import os
from datetime import datetime, timedelta

app = Flask(__name__)

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

# Flag to control video processing
stop_video_processing = False

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
                for _ in range(max_frames_to_capture):  # Corrected syntax error
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

# Video processing function with stop feature
def process_video(video_path):
    global stop_video_processing
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    stop_video_processing = False  # Reset stop flag when video starts

    while True:
        ret, frame = cap.read()
        if not ret or stop_video_processing:  # Stop processing if flag is set
            break

        frame = process_frame(frame)
        frame = resize_frame(frame)

        # Display the frame (Optional: remove if not needed)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('object_detection.html')

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
    process_video(video_path)
    print("Video detection started.")
    return "Video detection started."

@app.route('/stop_video_detection', methods=['GET'])
def stop_video_detection():
    global stop_video_processing
    stop_video_processing = True
    print("Video detection stopped.")
    return "Video detection stopped."

if __name__ == "__main__":
    app.run(debug=True)












