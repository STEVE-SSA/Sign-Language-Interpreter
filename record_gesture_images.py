import cv2
import os
import time
import mediapipe as mp
import numpy as np

# === USER CONFIGURATION ===
GESTURE_NAME = "sorry"      # Set the gesture name here
USER_ID = "Bhoomi"          # Set your user name here
NUM_SAMPLES = 50            # Number of images to record
HAND_MODE = 1              # 1 for single hand, 2 for double hand
CAPTURE_INTERVAL = 0.5      # Time in seconds between captures
# =========================

SAVE_DIR = f"datasets/custom_gestures/{GESTURE_NAME}"
os.makedirs(SAVE_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=HAND_MODE, min_detection_confidence=0.7)

IMG_SIZE = 480
BOX_SIZE = 300
PADDING = 20

# Central box
box_x = IMG_SIZE//2 - BOX_SIZE//2
box_y = IMG_SIZE//2 - BOX_SIZE//2

cap = cv2.VideoCapture(0)
cap.set(3, IMG_SIZE)
cap.set(4, IMG_SIZE)
print(f"\n🎥 Recording '{GESTURE_NAME}' gesture as {USER_ID}. Press 'q' to quit.")
print("Mode: {} hand(s)".format(HAND_MODE))

count = 0
last_capture_time = time.time()

while cap.isOpened() and count < NUM_SAMPLES:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    h, w, _ = frame.shape

    # Draw central box
    cv2.rectangle(frame, (box_x, box_y), (box_x+BOX_SIZE, box_y+BOX_SIZE), (0,255,0), 2)
    cv2.putText(frame, f"{GESTURE_NAME.upper()} | {USER_ID} | Sample {count+1}/{NUM_SAMPLES}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Mode: {HAND_MODE} Hand(s)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # Check for hands inside block
    hands_in_box = 0
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
            hx_min, hx_max = min(x_coords), max(x_coords)
            hy_min, hy_max = min(y_coords), max(y_coords)
            # Check if hand is inside the central box
            if hx_min > box_x+PADDING and hx_max < box_x+BOX_SIZE-PADDING and hy_min > box_y+PADDING and hy_max < box_y+BOX_SIZE-PADDING:
                hands_in_box += 1
    # Show status
    if hands_in_box == HAND_MODE:
        cv2.putText(frame, "Ready!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        current_time = time.time()
        if current_time - last_capture_time >= CAPTURE_INTERVAL:
            img_path = os.path.join(SAVE_DIR, f"{USER_ID}_{HAND_MODE}hand_{count+1:03d}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"📸 Captured: {img_path}")
            count += 1
            last_capture_time = current_time
    else:
        cv2.putText(frame, f"Place {HAND_MODE} hand(s) inside the box", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("Record Gesture", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
print(f"\n✅ Finished recording {count} samples for gesture '{GESTURE_NAME}' as {USER_ID}.")
