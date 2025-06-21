import io
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import mediapipe as mp

app = FastAPI()

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and label map
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "sequence_model_best.h5")
DATA_DIR = os.path.join(BASE_DIR, "sequence_data")
model = tf.keras.models.load_model(MODEL_PATH)
with open(os.path.join(DATA_DIR, "label_map.txt")) as f:
    label_map = {int(line.split(',')[0]): line.strip().split(',')[1] for line in f}
LABELS = [label_map[idx] for idx in sorted(label_map.keys())]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Feature extraction helpers (same as in data prep/inference)
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

# Buffer for sequence of features
from collections import deque
SEQ_LEN = 30  # Must match training and inference
buffer = deque(maxlen=SEQ_LEN)
N_KEYPOINTS = 2 * 21 * 3


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global buffer
    # Read and decode image
    contents = await file.read()
    img_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    frame = cv2.flip(frame, 1)  # Mirror to match OpenCV
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints = [0.0] * N_KEYPOINTS
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
        keypoints.extend([0.0] * (21 * 3))
    # Feature extraction (match infer_sequence_realtime.py)
    keypoints_np = np.array(keypoints).reshape(2, 21, 3)
    all_features = []
    for hand in keypoints_np:
        normed_hand = normalize_hand(hand)
        angles = [calculate_angle(normed_hand[idxs[0]], normed_hand[idxs[1]], normed_hand[idxs[2]]) for idxs in angle_indices]
        distances = calculate_distances(normed_hand)
        palm_orient = calculate_palm_orientation(normed_hand)
        all_features.extend(normed_hand.flatten())
        all_features.extend(angles)
        all_features.extend(distances)
        all_features.extend(palm_orient)
    buffer.append(np.array(all_features))
    # Temporal smoothing (moving average, window=3)
    if len(buffer) >= 3:
        arr = np.array(buffer)
        kernel = np.ones((3,))/3
        smoothed = np.vstack([np.convolve(arr[:,i], kernel, mode='same') for i in range(arr.shape[1])]).T
        buffer[-1] = smoothed[-1]
    # Only predict if buffer is full
    if len(buffer) == SEQ_LEN:
        input_seq = np.expand_dims(np.array(buffer), axis=0)
        preds = model.predict(input_seq)
        pred_label = LABELS[np.argmax(preds)]
        conf = float(np.max(preds))
        buffer.clear()  # Clear buffer after prediction, just like infer_sequence_realtime.py
    else:
        pred_label = None
        conf = None
    return JSONResponse({"prediction": pred_label, "confidence": conf, "buffer_len": len(buffer)})
