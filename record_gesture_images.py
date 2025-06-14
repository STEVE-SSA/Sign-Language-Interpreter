import cv2
import os
import time

# 🔧 Configuration
GESTURE_NAME = "help"  # 👈 Change this for each gesture
SAVE_DIR = f"datasets/custom_images/{GESTURE_NAME}/"
TOTAL_IMAGES = 50     # 📸 Total images to capture
CAPTURE_INTERVAL = 0.5  # 🕒 Seconds between captures (e.g., 0.5s = 2 fps)

# Create directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print(f"🎥 Starting capture for gesture: '{GESTURE_NAME}'")

count = 0
last_capture_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    now = time.time()

    # Time to capture new image
    if count < TOTAL_IMAGES and (now - last_capture_time) >= CAPTURE_INTERVAL:
        img_name = os.path.join(SAVE_DIR, f"{GESTURE_NAME}_{count}.jpg")
        cv2.imwrite(img_name, frame)
        last_capture_time = now
        count += 1
        capture_notation = True
    else:
        capture_notation = False

    # Add overlays
    if capture_notation:
        cv2.putText(frame, "📸 Capturing...", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(frame, f"Gesture: {GESTURE_NAME}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Images: {count}/{TOTAL_IMAGES}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Show frame
    cv2.imshow("Capture Gesture", frame)

    # Stop if ESC is pressed or capture complete
    if count >= TOTAL_IMAGES or cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Saved {count} images to: {SAVE_DIR}")
