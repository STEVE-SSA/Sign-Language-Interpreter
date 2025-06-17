import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
from collections import deque

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../gesture_model_best.h5'))
LABELS = sorted(os.listdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/train'))))
IMG_HEIGHT, IMG_WIDTH = 224, 448
STABILITY_FRAMES = 5
MOVEMENT_THRESH = 10

model = tf.keras.models.load_model(MODEL_PATH)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

prev_landmarks = deque(maxlen=STABILITY_FRAMES)
locked_label = None
locked_conf = None
waiting_for_still = True

cap = cv2.VideoCapture(0)
print("\n🖐️  Show your gesture to the camera. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    h, w, _ = frame.shape
    stable = False
    hand_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    landmarks_list = []
    # Detect hands and check stability
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            landmarks_list.append(landmarks)
        hand_imgs = []
        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
            y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
            pad = 20
            x_min, x_max = max(0, x_min-pad), min(w, x_max+pad)
            y_min, y_max = max(0, y_min-pad), min(h, y_max+pad)
            crop = frame[y_min:y_max, x_min:x_max]
            crop = cv2.resize(crop, (IMG_HEIGHT, IMG_HEIGHT))
            hand_imgs.append(crop)
        if len(hand_imgs) == 2:
            hand_img = np.concatenate([hand_imgs[0], hand_imgs[1]], axis=1)
        else:
            hand_img[:, :IMG_HEIGHT] = hand_imgs[0]
        prev_landmarks.append(landmarks_list)
        # Check stability only if not locked
        if locked_label is None and len(prev_landmarks) == STABILITY_FRAMES:
            mean_movement = 0
            for hand_idx in range(len(landmarks_list)):
                movement = 0
                for i in range(1, STABILITY_FRAMES):
                    prev = prev_landmarks[i-1][hand_idx] if hand_idx < len(prev_landmarks[i-1]) else None
                    curr = prev_landmarks[i][hand_idx] if hand_idx < len(prev_landmarks[i]) else None
                    if prev and curr:
                        movement += np.mean([np.linalg.norm(np.array(p)-np.array(c)) for p, c in zip(prev, curr)])
                movement /= (STABILITY_FRAMES-1)
                mean_movement += movement
            mean_movement /= len(landmarks_list)
            if mean_movement < MOVEMENT_THRESH:
                stable = True
        # If stable and not locked, predict and lock
        if stable and locked_label is None:
            input_img = hand_img.astype(np.float32) / 255.0
            input_img = np.expand_dims(input_img, axis=0)
            preds = model.predict(input_img)
            locked_label = LABELS[np.argmax(preds)]
            locked_conf = np.max(preds)
            waiting_for_still = False
        # If locked, check for movement to reset
        elif locked_label is not None and len(prev_landmarks) == STABILITY_FRAMES:
            mean_movement = 0
            for hand_idx in range(len(landmarks_list)):
                movement = 0
                for i in range(1, STABILITY_FRAMES):
                    prev = prev_landmarks[i-1][hand_idx] if hand_idx < len(prev_landmarks[i-1]) else None
                    curr = prev_landmarks[i][hand_idx] if hand_idx < len(prev_landmarks[i]) else None
                    if prev and curr:
                        movement += np.mean([np.linalg.norm(np.array(p)-np.array(c)) for p, c in zip(prev, curr)])
                movement /= (STABILITY_FRAMES-1)
                mean_movement += movement
            mean_movement /= len(landmarks_list)
            if mean_movement > MOVEMENT_THRESH * 2:  # Require significant movement to reset
                locked_label = None
                locked_conf = None
                waiting_for_still = True
                prev_landmarks.clear()
    else:
        prev_landmarks.clear()
        locked_label = None
        locked_conf = None
        waiting_for_still = True
    # Display
    if locked_label is not None:
        cv2.putText(frame, f"{locked_label} ({locked_conf:.2f})", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, "Move hand to reset", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "Waiting for stable gesture...", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imshow("Sign Language Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
hands.close()
cv2.destroyAllWindows()
