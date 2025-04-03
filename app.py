import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory, Response
from werkzeug.utils import secure_filename

# Load trained CNN model
MODEL_PATH = "face_mask_detector.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Initialize Flask app
app = Flask(__name__)

# Define Upload Folder Path
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Image size for model input
IMG_SIZE = 128  

# Function to preprocess and predict mask status
def predict_mask(image):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = np.array(image, dtype=np.float32) / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  

        prediction = model.predict(image)[0][0]
        
        label = "Mask Detected" if prediction < 0.5 else "No Mask Detected"
        color = (0, 255, 0) if label == "Mask Detected" else (0, 0, 255)  # Green for mask, red for no mask

        return label, color
    except Exception as e:
        print("Error in prediction:", e)
        return "Error", (0, 0, 0)

# Route to handle file uploads
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded!"
        
        file = request.files["file"]

        if file.filename == "":
            return "No selected file!"
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            image = cv2.imread(filepath)
            label, _ = predict_mask(image)

            return render_template("index.html", filename=filename, label=label)

    return render_template("index.html")

# Route to display uploaded image
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# Live Camera Detection
def generate_frames():
    cap = cv2.VideoCapture(0)  # Open webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Face detection using OpenCV
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                label, color = predict_mask(face)

                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Encode frame
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

# Route for live camera feed
@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# Route to render the live detection page
@app.route("/live")
def live():
    return render_template("live.html")

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
