import cv2
import numpy as np

def preprocess_image(image):
    # Resize and normalize the image (ensure it matches model input dimensions)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def predict_face(model, image, label_classes):
    # Predict the class and retrieve the label
    prediction = model.predict(image)
    predicted_index = np.argmax(prediction)
    name, batch = label_classes[predicted_index].split('_')
    return name, batch
