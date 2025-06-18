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
            X.append(np.load(seq_path))
            y.append(idx)
X = np.array(X)  # shape: (num_samples, seq_len, 126)
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
