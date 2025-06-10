import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Paths
DATASET_CSV = 'datasets/sign_mnist/sign_mnist_train.csv'
OUTPUT_CSV = 'datasets/sign_mnist_landmarks.csv'

# Load dataset
df = pd.read_csv(DATASET_CSV)
all_landmarks = []

print("Processing images and extracting landmarks...")

for idx, row in df.iterrows():
    if idx > 5000 :
        break
    label = row[0]
    pixels = np.array(row[1:]).reshape(28, 28).astype(np.uint8)

    # Convert to 3-channel image (MediaPipe requires RGB)
    image = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)

    # Resize to 224x224 for better detection
    image = cv2.resize(image, (224, 224))
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_row = [label]
            for lm in hand_landmarks.landmark:
                landmark_row.extend([lm.x, lm.y, lm.z])
            all_landmarks.append(landmark_row)
    else:
        continue  # Skip images where no hand was detected

    if idx % 500 == 0:
        print(f"Processed {idx} images")

# Save to CSV
columns = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ['_x', '_y', '_z']]
landmarks_df = pd.DataFrame(all_landmarks, columns=columns)
landmarks_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nDone! Extracted landmarks saved to: {OUTPUT_CSV}")
