import cv2
import mediapipe as mp
import pandas as pd
import os

# 🔧 Customize here
LABEL = "thankyou"  # ⬅️ Change to your gesture name
SAMPLES = 200   # ⬅️ Number of samples to record
SAVE_PATH = f"datasets/custom_landmarks/{LABEL}.csv"

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)
print(f"🎥 Recording '{LABEL}' gesture... Show it on camera.")

data = []
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm_row = []
            for lm in hand_landmarks.landmark:
                lm_row.extend([lm.x, lm.y, lm.z])
            data.append([LABEL] + lm_row)
            count += 1

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Samples: {count}/{SAMPLES}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Recording Gesture", frame)

    if count >= SAMPLES or cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Save CSV
columns = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ['_x', '_y', '_z']]
df = pd.DataFrame(data, columns=columns)
df.to_csv(SAVE_PATH, index=False)

print(f"\n✅ Saved {count} samples to: {SAVE_PATH}")
