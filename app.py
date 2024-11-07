from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
from utils import preprocess_image, predict_face

# Initialize the Flask app and load the model
app = Flask(__name__)
model = tf.keras.models.load_model('./models/face_recognition_model.h5')

# Initialize the label encoder to map predicted labels back to names and batches
label_classes = np.load('label_classes.npy')

# Define the video capture object (using the webcam)
camera = cv2.VideoCapture(0)

# Function to generate frames from the webcam for the web app
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Preprocess the frame and make a prediction
            processed_frame = preprocess_image(frame)
            prediction = predict_face(model, processed_frame, label_classes)
            
            # Display the predicted name and batch on the frame
            label_text = f"Name: {prediction[0]}, Batch: {prediction[1]}"
            cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Define the Flask route to stream the video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Define the home route to display the HTML template
@app.route('/')
def index():
    return render_template('index.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
