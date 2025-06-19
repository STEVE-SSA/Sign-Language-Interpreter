import cv2
import os
import time
import mediapipe as mp
import numpy as np
import json

# === USER CONFIGURATION ===
GESTURE_NAME = "I"      # Set the gesture name here
USER_ID = "Steve"            # Set your user name here
NUM_SEQUENCES = 30             # Number of sequences to record
SEQUENCE_LENGTH = 30           # Frames per sequence
SAVE_DIR = f"datasets/sequences/{GESTURE_NAME}"
# =========================

os.makedirs(SAVE_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

IMG_SIZE = 480
BOX_SIZE = 300
PADDING = 20

box_x = IMG_SIZE//2 - BOX_SIZE//2
box_y = IMG_SIZE//2 - BOX_SIZE//2

cap = cv2.VideoCapture(0)
cap.set(3, IMG_SIZE)
cap.set(4, IMG_SIZE)

print(f"\n🎥 Recording gesture '{GESTURE_NAME}' as {USER_ID}. Press 'q' to quit.")
print(f"Each sequence will be {SEQUENCE_LENGTH} frames long. {NUM_SEQUENCES} sequences will be recorded.")

seq_count = 0
while seq_count < NUM_SEQUENCES:
    sequence = []
    frame_count = 0
    print(f"\nGet ready for sequence {seq_count+1}/{NUM_SEQUENCES}...")
    time.sleep(1.5)
    while frame_count < SEQUENCE_LENGTH:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        h, w, _ = frame.shape
        # Draw central box
        cv2.rectangle(frame, (box_x, box_y), (box_x+BOX_SIZE, box_y+BOX_SIZE), (0,255,0), 2)
        cv2.putText(frame, f"Gesture: {GESTURE_NAME} | User: {USER_ID}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Sequence: {seq_count+1}/{NUM_SEQUENCES} Frame: {frame_count+1}/{SEQUENCE_LENGTH}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        keypoints = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
        else:
            # If no hand detected, fill with zeros (same length as 2 hands x 21 keypoints x 3)
            keypoints = [0.0] * (2 * 21 * 3)
        # Pad if only one hand detected
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
            keypoints.extend([0.0] * (21 * 3))
        sequence.append(keypoints)
        cv2.imshow("Record Gesture Sequence", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
        frame_count += 1
    # Save sequence as .npy
    seq_path = os.path.join(SAVE_DIR, f"{USER_ID}_seq{seq_count+1:02d}.npy")
    np.save(seq_path, np.array(sequence))
    print(f"✅ Saved sequence {seq_count+1}: {seq_path}")
    seq_count += 1
print(f"\nAll sequences for gesture '{GESTURE_NAME}' recorded!")
cap.release()
hands.close()
cv2.destroyAllWindows()
