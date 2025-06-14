import cv2
import numpy as np
import tensorflow as tf
import time
import os

# Load model and label map
model = tf.keras.models.load_model("gesture_cnn_model.h5")
label_map = np.load("gesture_data.npz", allow_pickle=True)["label_map"].item()
reverse_label_map = {v: k for k, v in label_map.items()}

IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.85
STABILITY_SECONDS = 1.2  # How long a gesture must stay stable

# Initialize webcam
cap = cv2.VideoCapture(0)
prev_prediction = None
stable_since = None

print("📷 Show a gesture in front of the webcam...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    roi = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    roi = roi / 255.0
    roi = np.expand_dims(roi, axis=0)

    prediction = model.predict(roi)[0]
    confidence = np.max(prediction)
    predicted_class = np.argmax(prediction)
    label = reverse_label_map[predicted_class]

    # Gesture Stability Logic
    if confidence > CONFIDENCE_THRESHOLD:
        if prev_prediction == label:
            if stable_since and time.time() - stable_since >= STABILITY_SECONDS:
                text = f"{label.upper()}"
            else:
                text = "⏳ Hold gesture..."
                if not stable_since:
                    stable_since = time.time()
        else:
            prev_prediction = label
            stable_since = time.time()
            text = "⏳ Hold gesture..."
    else:
        text = "❌ No confident gesture"
        prev_prediction = None
        stable_since = None

    # Overlay prediction text
    cv2.putText(frame, text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Gesture Recognition", frame)

    # Exit with ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
