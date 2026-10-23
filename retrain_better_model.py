#!/usr/bin/env python3
"""
Retrain the model with better architecture and more epochs
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os
from data_split import make_generators

# Fix for Unicode issues
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Paths
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive/colored_images")
model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive/my_saved_model_improved.h5")

# Parameters
img_height, img_width = 224, 224
batch_size = 32
epochs = int(os.environ.get("EPOCHS", 50))  # More epochs (override with EPOCHS env var)
num_classes = 5

# Enhanced Image Generators with more augmentation (training only)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
# Validation images are only rescaled, so val metrics are stable between epochs
val_datagen = ImageDataGenerator(rescale=1./255)

# Split per original image so balance_dataset.py copies can't leak into validation
train_generator, validation_generator = make_generators(
    data_dir, train_datagen, val_datagen,
    target_size=(img_height, img_width), batch_size=batch_size
)

# Print class mapping
print("Class mapping:")
for class_name, idx in train_generator.class_indices.items():
    print(f"{class_name} -> {idx}")

# Compute class weights
classes = list(train_generator.class_indices.keys())
labels = train_generator.classes
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights_dict = dict(enumerate(class_weights))

print(f"\nClass weights: {class_weights_dict}")
print(f"Class distribution: {np.bincount(labels)}")

# Improved Model Architecture
model = Sequential([
    tf.keras.Input(shape=(img_height, img_width, 3)),
    
    # First block
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.25),
    
    # Second block
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.25),
    
    # Third block
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.25),
    
    # Fourth block
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.25),
    
    # Dense layers
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile with better optimizer
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Optionally resume from a saved checkpoint (RESUME_FROM=path, LR=learning rate,
# BEST_VAL_ACC=previous best so the checkpoint isn't overwritten by a worse epoch)
resume_path = os.environ.get("RESUME_FROM")
best_val_acc = float(os.environ["BEST_VAL_ACC"]) if "BEST_VAL_ACC" in os.environ else None
if resume_path:
    print(f"\nResuming from {resume_path}")
    model = tf.keras.models.load_model(resume_path)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(os.environ.get("LR", 0.001))),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

print("\nModel summary:")
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1,
                    initial_value_threshold=best_val_acc),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

# Train
print("\nStarting training...")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    class_weight=class_weights_dict,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
print("\nEvaluating model...")
validation_generator.reset()
preds = model.predict(validation_generator, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = validation_generator.classes[:len(y_pred)]

print('\nConfusion Matrix:')
print(confusion_matrix(y_true, y_pred))
print('\nClassification Report:')
print(classification_report(y_true, y_pred, target_names=classes))

# Save final model separately so the best-val_accuracy checkpoint isn't overwritten
final_model_path = model_save_path.replace(".h5", "_final.h5")
try:
    model.save(final_model_path)
    print(f"✅ Final model saved to {final_model_path} (best checkpoint: {model_save_path})")
except Exception as e:
    print(f"❌ Failed to save model: {e}")

print("\nTraining completed! Use the improved model in your app.")
