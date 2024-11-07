# preprocess.py
import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Define data directory
DATA_DIR = './dataset'  # Ensure this points to your main dataset directory
IMAGE_SIZE = (128, 128)  # Set the target size for each image

# Function to load and preprocess images
def load_images(dataset_dir):
    images = []
    labels = []
    
    # List each person's folder (e.g., Aavani_DSA2024, Bharath_DSA2024, Hafsana_DSA2024)
    for label in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, label)
        
        if os.path.isdir(person_dir):  # Process only folders
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                image = cv2.imread(image_path)
                
                if image is not None:
                    # Resize and normalize the image
                    image = cv2.resize(image, IMAGE_SIZE)
                    image = image / 255.0  # Normalize to range [0, 1]
                    images.append(image)
                    labels.append(label)  # Use the folder name as the label
                    
    return np.array(images), np.array(labels)

# Load and preprocess images
images, labels = load_images(DATA_DIR)

# Encode labels into numeric format for training
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Save processed data for model training
np.save('images.npy', images)
np.save('labels.npy', encoded_labels)
np.save('label_classes.npy', label_encoder.classes_)

print("Data preprocessing completed successfully.")
