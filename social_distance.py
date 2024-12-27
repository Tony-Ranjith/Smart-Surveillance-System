from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
from gtts import gTTS
from playsound import playsound
import os

app = Flask(__name__)

# Globals for live detection control and video file path
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
@app.route('/')
def index():
    return render_template('social_distance.html')

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

if __name__== "__main__":
    app.run(debug=True)