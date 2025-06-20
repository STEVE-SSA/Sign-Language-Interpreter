import numpy as np
import tensorflow as tf
import os

# Load data
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sequence_data'))
X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

with open(os.path.join(DATA_DIR, 'label_map.txt')) as f:
    label_map = {int(line.split(',')[0]): line.strip().split(',')[1] for line in f}
num_classes = len(label_map)
seq_len = X_train.shape[1]
input_dim = X_train.shape[2]

# Model definition (LSTM-based)
model = tf.keras.Sequential([
    tf.keras.layers.Masking(mask_value=0.0, input_shape=(seq_len, input_dim)),  # input_dim now includes angles
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('sequence_model_best.h5', save_best_only=True)
]

print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=callbacks
)

print("\nEvaluating on test set...")
loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.3f}")
