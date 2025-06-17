import os
import cv2
import mediapipe as mp
import numpy as np
import shutil
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# CONFIGURABLES
DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/custom_gestures'))
OUTPUT_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.15, 0.15
IMG_SIZE = 224  # For MobileNetV2, EfficientNet, etc.
AUGMENTATIONS_PER_IMAGE = 3

mp_hands = mp.solutions.hands

# Augmentation functions
def augment_image(img):
    aug_imgs = []
    # Horizontal flip
    aug_imgs.append(cv2.flip(img, 1))
    # Random brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    value = random.randint(-30, 30)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
    aug_imgs.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
    # Random rotation
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((IMG_SIZE//2, IMG_SIZE//2), angle, 1)
    aug_imgs.append(cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE)))
    return aug_imgs

def is_blurry(img, thresh=100.0):
    # Variance of Laplacian for blur detection
    return cv2.Laplacian(img, cv2.CV_64F).var() < thresh

# Prepare output directories
def prepare_dirs():
    if os.path.exists(OUTPUT_BASE):
        shutil.rmtree(OUTPUT_BASE)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_BASE, split), exist_ok=True)

def crop_hands(image, results):
    h, w, _ = image.shape
    hand_imgs = []
    for hand_landmarks in results.multi_hand_landmarks:
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
        y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
        # Padding
        pad = 20
        x_min, x_max = max(0, x_min-pad), min(w, x_max+pad)
        y_min, y_max = max(0, y_min-pad), min(h, y_max+pad)
        hand_imgs.append(image[y_min:y_max, x_min:x_max])
    return hand_imgs

def process_and_save():
    prepare_dirs()
    all_data = []
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7)
    gesture_classes = os.listdir(DATASET_DIR)
    for gesture in tqdm(gesture_classes, desc='Gestures'):
        gesture_dir = os.path.join(DATASET_DIR, gesture)
        if not os.path.isdir(gesture_dir):
            continue
        images = [f for f in os.listdir(gesture_dir) if f.lower().endswith('.jpg')]
        for img_name in images:
            img_path = os.path.join(gesture_dir, img_name)
            img = cv2.imread(img_path)
            if img is None or is_blurry(img):
                continue  # Skip blurry or unreadable images
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            if not results.multi_hand_landmarks:
                continue  # No hand detected
            hand_imgs = crop_hands(img, results)
            # For both hands, concatenate side by side
            if len(hand_imgs) == 2:
                hand_img = np.concatenate([cv2.resize(h, (IMG_SIZE, IMG_SIZE)) for h in hand_imgs], axis=1)
            else:
                hand_img = cv2.resize(hand_imgs[0], (IMG_SIZE, IMG_SIZE))
            all_data.append((hand_img, gesture))
            # Augmentations
            for aug_img in augment_image(hand_img):
                all_data.append((aug_img, gesture))
    hands.close()
    # Shuffle and split
    random.shuffle(all_data)
    X = [x[0] for x in all_data]
    y = [x[1] for x in all_data]
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=(1-TRAIN_RATIO), stratify=y, random_state=42)
    val_size = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=val_size, stratify=y_tmp, random_state=42)
    # Save images
    for split, X_split, y_split in zip(['train','val','test'], [X_train, X_val, X_test], [y_train, y_val, y_test]):
        for i, (img, label) in enumerate(zip(X_split, y_split)):
            out_dir = os.path.join(OUTPUT_BASE, split, label)
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, f'{label}_{i:04d}.jpg'), img)
    print('✅ Data preparation complete. Train/val/test sets saved to:', OUTPUT_BASE)

if __name__ == '__main__':
    process_and_save()
