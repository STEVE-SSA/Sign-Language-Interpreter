import os
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

# Path to your sequence data
SEQUENCE_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/sequences'))
OUTPUT_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sequence_data'))

# Clean output directory
if os.path.exists(OUTPUT_BASE):
    shutil.rmtree(OUTPUT_BASE)
os.makedirs(OUTPUT_BASE, exist_ok=True)

X = []  # sequences
y = []  # labels
label_map = {}
label_list = sorted(os.listdir(SEQUENCE_DATA_DIR))
for idx, gesture in enumerate(label_list):
    label_map[gesture] = idx
    gesture_dir = os.path.join(SEQUENCE_DATA_DIR, gesture)
    if not os.path.isdir(gesture_dir):
        continue
    for fname in os.listdir(gesture_dir):
        if fname.endswith('.npy'):
            seq_path = os.path.join(gesture_dir, fname)
            seq = np.load(seq_path)  # shape: (seq_len, 126)
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
            seq_features = []
            for frame in seq:
                keypoints = frame.reshape(2, 21, 3)
                all_features = []
                for hand in keypoints:
                    normed_hand = normalize_hand(hand)
                    angles = [calculate_angle(normed_hand[idxs[0]], normed_hand[idxs[1]], normed_hand[idxs[2]]) for idxs in angle_indices]
                    distances = calculate_distances(normed_hand)
                    palm_orient = calculate_palm_orientation(normed_hand)
                    all_features.extend(normed_hand.flatten())
                    all_features.extend(angles)
                    all_features.extend(distances)
                    all_features.extend(palm_orient)
                seq_features.append(np.array(all_features))
            seq_features = np.array(seq_features)
            if seq_features.shape[0] >= 3:
                kernel = np.ones((3,1))/3
                seq_features = np.vstack([np.convolve(seq_features[:,i], kernel[:,0], mode='same') for i in range(seq_features.shape[1])]).T
            X.append(seq_features)
            y.append(idx)
X = np.array(X)  # shape: (num_samples, seq_len, 126 + 20)
y = np.array(y)

# Split into train/val/test
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

# Save splits
np.save(os.path.join(OUTPUT_BASE, 'X_train.npy'), X_train)
np.save(os.path.join(OUTPUT_BASE, 'y_train.npy'), y_train)
np.save(os.path.join(OUTPUT_BASE, 'X_val.npy'), X_val)
np.save(os.path.join(OUTPUT_BASE, 'y_val.npy'), y_val)
np.save(os.path.join(OUTPUT_BASE, 'X_test.npy'), X_test)
np.save(os.path.join(OUTPUT_BASE, 'y_test.npy'), y_test)
with open(os.path.join(OUTPUT_BASE, 'label_map.txt'), 'w') as f:
    for gesture, idx in label_map.items():
        f.write(f'{idx},{gesture}\n')
print('✅ Sequence data prepared and saved to', OUTPUT_BASE)
