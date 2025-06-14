import cv2
import os
import time

# === CUSTOMIZE HERE ===
GESTURE_NAME = "help"      # Change this for each gesture
USER_ID = "Bhoomi"          # Change to "user2" for second person
NUM_SAMPLES = 50
CAPTURE_INTERVAL = 0.5     # Time in seconds between captures
SAVE_DIR = f"datasets/custom_gestures/{GESTURE_NAME}"
# =======================

# Create folder if not exists
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
print(f"🎥 Recording '{GESTURE_NAME}' gesture as {USER_ID}. Show it on camera.")

count = 0
last_capture_time = time.time()

while cap.isOpened() and count < NUM_SAMPLES:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    cv2.putText(frame, f"{GESTURE_NAME.upper()} | {USER_ID} | Sample {count+1}/{NUM_SAMPLES}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Record Gesture", frame)

    # Capture image automatically based on time interval
    current_time = time.time()
    if current_time - last_capture_time >= CAPTURE_INTERVAL:
        img_path = os.path.join(SAVE_DIR, f"{USER_ID}_{count+1:03d}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"📸 Captured: {img_path}")
        count += 1
        last_capture_time = current_time

    # Press ESC to exit early
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Saved {count} images to: {SAVE_DIR}")
