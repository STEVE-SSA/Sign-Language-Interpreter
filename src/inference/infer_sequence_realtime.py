import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
from collections import deque

# Load model and label map
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sequence_model_best.h5'))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sequence_data'))
with open(os.path.join(DATA_DIR, 'label_map.txt')) as f:
    label_map = {int(line.split(',')[0]): line.strip().split(',')[1] for line in f}
LABELS = [label_map[idx] for idx in sorted(label_map.keys())]

model = tf.keras.models.load_model(MODEL_PATH)

# Parameters
SEQ_LEN = 30  # Must match training
N_KEYPOINTS = 2 * 21 * 3

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
IMG_SIZE = 480
BOX_SIZE = 300
box_x = IMG_SIZE//2 - BOX_SIZE//2
box_y = IMG_SIZE//2 - BOX_SIZE//2

buffer = deque(maxlen=SEQ_LEN)
last_pred = None
last_conf = 0

import time

print("\n🖐️  Get ready to show your gesture. Press 'q' to quit.")

# --- Loop for multiple gesture detection ---
COUNTDOWN = 3
DISPLAY_TIME = 10  # seconds

while cap.isOpened():
    # Countdown
    for i in range(COUNTDOWN, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        cv2.rectangle(frame, (box_x, box_y), (box_x+BOX_SIZE, box_y+BOX_SIZE), (0,255,0), 2)
        cv2.putText(frame, f"Starting in {i}...", (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
        cv2.imshow("Sequence Gesture Inference", frame)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            cap.release()
            hands.close()
            cv2.destroyAllWindows()
            exit()
    # Record one gesture sequence
    buffer.clear()
    recording = True
    predicted = False
    while cap.isOpened() and recording:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        h, w, _ = frame.shape
        cv2.rectangle(frame, (box_x, box_y), (box_x+BOX_SIZE, box_y+BOX_SIZE), (0,255,0), 2)
        keypoints = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints = [0.0] * N_KEYPOINTS
        # Pad if only one hand detected
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
            keypoints.extend([0.0] * (21 * 3))
        # --- Advanced feature extraction for real-time inference ---
        def calculate_angle(a, b, c):
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return np.degrees(angle)
        def calculate_distances(hand):
            wrist = hand[0]
            fingertips = [hand[i] for i in [4,8,12,16,20]]
            return [np.linalg.norm(f - wrist) for f in fingertips]
        def calculate_palm_orientation(hand):
            v1 = hand[5] - hand[0]
            v2 = hand[17] - hand[0]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            return normal / (norm + 1e-6)
        def normalize_hand(hand):
            wrist = hand[0]
            normed = hand - wrist
            scale = np.linalg.norm(hand[12] - wrist) + 1e-6
            return normed / scale
        angle_indices = [
            (0, 1, 2), (1, 2, 3), (2, 3, 4),
            (0, 5, 6), (5, 6, 7), (6, 7, 8),
            (0, 9, 10), (9, 10, 11), (10, 11, 12),
            (0, 13, 14), (13, 14, 15), (14, 15, 16),
            (0, 17, 18), (17, 18, 19), (18, 19, 20)
        ]
        keypoints_np = np.array(keypoints).reshape(2, 21, 3)
        all_features = []
        for hand in keypoints_np:
            normed_hand = normalize_hand(hand)
            # Angles
            angles = [calculate_angle(normed_hand[idxs[0]], normed_hand[idxs[1]], normed_hand[idxs[2]]) for idxs in angle_indices]
            # Distances
            distances = calculate_distances(normed_hand)
            # Palm orientation
            palm_orient = calculate_palm_orientation(normed_hand)
            # Concatenate all features for this hand
            all_features.extend(normed_hand.flatten())
            all_features.extend(angles)
            all_features.extend(distances)
            all_features.extend(palm_orient)
        buffer.append(np.array(all_features))
        # --- Temporal smoothing: apply moving average to buffer if enough frames ---
        if len(buffer) >= 3:
            arr = np.array(buffer)
            kernel = np.ones((3,))/3
            smoothed = np.vstack([np.convolve(arr[:,i], kernel, mode='same') for i in range(arr.shape[1])]).T
            # Replace last frame with its smoothed version
            buffer[-1] = smoothed[-1]

        cv2.putText(frame, f"Recording gesture... ({len(buffer)}/{SEQ_LEN})", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        cv2.imshow("Sequence Gesture Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            hands.close()
            cv2.destroyAllWindows()
            exit()
        if len(buffer) == SEQ_LEN:
            recording = False
            predicted = True
            break
    # Predict and show result for DISPLAY_TIME seconds
    if predicted and len(buffer) == SEQ_LEN:
        input_seq = np.expand_dims(np.array(buffer), axis=0)
        preds = model.predict(input_seq)
        pred_label = LABELS[np.argmax(preds)]
        conf = np.max(preds)
        start_time = time.time()
        while time.time() - start_time < DISPLAY_TIME:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            cv2.rectangle(frame, (box_x, box_y), (box_x+BOX_SIZE, box_y+BOX_SIZE), (0,255,0), 2)
            cv2.putText(frame, f"Detected: {pred_label} ({conf:.2f})", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, "Next gesture in a moment... (or press 'q' to quit)", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
            cv2.imshow("Sequence Gesture Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                hands.close()
                cv2.destroyAllWindows()
                exit()
# Cleanup
cap.release()
hands.close()
cv2.destroyAllWindows()
