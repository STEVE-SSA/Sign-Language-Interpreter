import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
MODEL_SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../gesture_model_best.h5'))
IMG_HEIGHT, IMG_WIDTH = 224, 448  # Both hands side-by-side
BATCH_SIZE = 16
EPOCHS = 30

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8,1.2],
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

def get_generator(split):
    dir_path = os.path.join(DATA_DIR, split)
    return train_datagen.flow_from_directory(
        dir_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True if split=='train' else False
    )

train_gen = get_generator('train')
val_gen = get_generator('val')
num_classes = train_gen.num_classes

# Model
base_model = MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Fine-tune later if needed

inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs, outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

print(f"\n✅ Training complete. Best model saved to: {MODEL_SAVE_PATH}")
