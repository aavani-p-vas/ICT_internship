# train_model.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load the preprocessed data
images = np.load('images.npy')
labels = np.load('labels.npy')
label_classes = np.load('label_classes.npy')

# Get input shape and the number of classes
input_shape = images.shape[1:]  # (128, 128, 3)
num_classes = len(label_classes)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# One-hot encode the labels for training
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)

# Define the CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Softmax for multi-class classification
    ])
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize the CNN model
model = create_cnn_model(input_shape, num_classes)
model.summary()  # Print model architecture

# Train the model
history = model.fit(X_train, y_train, epochs=15, validation_data=(X_val, y_val), batch_size=8)

# Save the trained model
model.save('./models/face_recognition_model.h5')
print("Model training completed and saved as face_recognition_model.h5")
