from flask import Flask, render_template, Response, request, jsonify
import numpy as np
import cv2
import os
from collections import deque
import imutils

app = Flask(__name__)

# Configuration
MODEL_PATH = "resnet-34_kinetics.onnx"  # Change this to your actual model path
CLASSES_PATH = "Actions.txt"  # Change this to your actual class labels file
SAMPLE_DURATION = 16
SAMPLE_SIZE = 112
frames = deque(maxlen=SAMPLE_DURATION)

# Load the model and class labels
print("[INFO] Loading model...")
net = cv2.dnn.readNet(MODEL_PATH)
CLASSES = open(CLASSES_PATH).read().strip().split("\n")

# Function to capture live video feed
def generate_frames():
    cap = cv2.VideoCapture(0)  # Use webcam (Change to 1 or 2 if needed)
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame = imutils.resize(frame, width=400)
        frames.append(frame)

        if len(frames) == SAMPLE_DURATION:
            blob = cv2.dnn.blobFromImages(frames, 1.0, (SAMPLE_SIZE, SAMPLE_SIZE),
                                          (114.7748, 107.7354, 99.4750), swapRB=True, crop=True)
            blob = np.transpose(blob, (1, 0, 2, 3))
            blob = np.expand_dims(blob, axis=0)
            net.setInput(blob)
            outputs = net.forward()
            label = CLASSES[np.argmax(outputs)]

            cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Route for live streaming video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
