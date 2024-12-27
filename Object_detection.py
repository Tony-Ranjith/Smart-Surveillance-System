import cv2
import numpy as np
import telebot
import time
import os
from datetime import datetime, timedelta

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
            # Check if the file is older than 1 hour
            if current_time - file_mod_time > timedelta(hours=hours):
                os.remove(file_path)
                print(f"Deleted old frame: {file_path}")

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

def send_telegram_photo(photo):
    try:
        with open(photo, 'rb') as f:
            bot.send_photo(chat_id=CHAT_ID, photo=f, caption="A moving object is detected!")
        print(f"Sent photo to Telegram: {photo}")
    except Exception as e:
        print(f"Failed to send photo: {e}")

def process_frame(frame):
    global currently_detected_objects

    # Apply background subtraction to detect movement
    fg_mask = back_sub.apply(frame)
    _, fg_mask_thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)  # Remove shadows and noise

    # Check if any movement is detected
    motion_detected = np.sum(fg_mask_thresh) > 20000  # Adjust the threshold as needed

    indexes, boxes, confidences, class_ids = detect_objects(frame)
    detected_this_frame = []

    if len(indexes) > 0 and motion_detected:  # Only proceed if motion is detected
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            object_id = f"{class_ids[i]}_{x}_{y}_{w}_{h}"  # Unique ID based on class and bounding box

            # Track the detection
            if object_id not in currently_detected_objects:
                currently_detected_objects[object_id] = {
                    'last_seen': time.time(),
                    'frame_count': 0,
                    'notified': False
                }

            # Capture frames if object is newly detected or has moved
            if not currently_detected_objects[object_id]['notified']:
                for _ in range(max_frames_to_capture):  # Capture three frames
                    frame_name = f"{object_id}_{time.time()}.jpg"
                    frame_path = os.path.join(object_detected_folder, frame_name)
                    cv2.imwrite(frame_path, frame)  # Save the frame
                    print(f"Frame saved as: {frame_path}")

                    send_telegram_photo(frame_path)  # Send the frame

                # Mark as notified and update last seen time
                currently_detected_objects[object_id]['notified'] = True
                currently_detected_objects[object_id]['last_seen'] = time.time()

            detected_this_frame.append(object_id)

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw boundary
            cv2.putText(frame, f'ID: {class_ids[i]}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Remove objects not seen for a while
    for object_id in list(currently_detected_objects.keys()):
        if object_id not in detected_this_frame:
            currently_detected_objects[object_id]['notified'] = False  # Reset notification flag
            currently_detected_objects.pop(object_id)

    return frame

def resize_frame(frame, width=800):
    aspect_ratio = frame.shape[0] / frame.shape[1]
    new_height = int(width * aspect_ratio)
    return cv2.resize(frame, (width, new_height))

def real_time_detection():
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
        frame = resize_frame(frame)  # Resize the frame for better display
        
        # Display the video feed in an external window
        cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)  # Open the window in normal mode
        cv2.imshow("Webcam", frame)  # Show the frame in the window
        cv2.setWindowProperty("Webcam", cv2.WND_PROP_TOPMOST, 1)  # Set window as topmost
        
        # Check for old frames to delete every minute
        if int(time.time()) % 60 == 0:
            delete_old_frames(object_detected_folder)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from video.")
            break
        frame = process_frame(frame)
        frame = resize_frame(frame)  # Resize the frame for better display
        
        # Display the video feed in an external window
        cv2.namedWindow("Video", cv2.WINDOW_NORMAL)  # Open the window in normal mode
        cv2.imshow("Video", frame)  # Show the frame in the window
        cv2.setWindowProperty("Video", cv2.WND_PROP_TOPMOST, 1)  # Set window as topmost
        
        # Check for old frames to delete every minute
        if int(time.time()) % 60 == 0:
            delete_old_frames(object_detected_folder)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mode = input("Enter '1' for Real-time detection or '2' for processing a local video: ")
    if mode == '1':
        real_time_detection()
    elif mode == '2':
        video_path = input("Enter the path to the local video file: ")
        process_video(video_path)
    else:
        print("Invalid option. Please enter '1' or '2'.")

















































