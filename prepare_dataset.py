import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# CONFIG
IMG_SIZE = 128
DATASET_PATH = "datasets/custom_gestures"

# Label mapping
labels = sorted(os.listdir(DATASET_PATH))
label_map = {label: idx for idx, label in enumerate(labels)}

# Load images
X, y = [], []

for label in labels:
    folder_path = os.path.join(DATASET_PATH, label)
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(label_map[label])

X = np.array(X) / 255.0  # Normalize
y = to_categorical(y, num_classes=len(labels))

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save for reuse
np.savez("gesture_data.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, label_map=label_map)
print("✅ Dataset prepared and saved.")
