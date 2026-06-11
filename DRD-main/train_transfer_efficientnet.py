import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report


# Fix stdout encoding on some environments
sys.stdout.reconfigure(encoding='utf-8')


# Paths
DATA_DIR = "/Users/pradeepkd/Desktop/projectphase1/archive/colored_images"
MODEL_PATH = "/Users/pradeepkd/Desktop/projectphase1/archive/my_saved_model_efficientnet.h5"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


# Params
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50


# Data Generators
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)


# Class weights
classes = list(train_gen.class_indices.keys())
labels = train_gen.classes
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights_dict = dict(enumerate(class_weights))


# Model
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=Input(shape=(*IMG_SIZE, 3)))
base_model.trainable = False  # first stage: freeze

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
outputs = Dense(train_gen.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])


# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]


# Train - stage 1
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights_dict,
    callbacks=callbacks
)


# Fine-tune - unfreeze some top layers
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

history_ft = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights_dict,
    callbacks=callbacks
)


# Evaluate
val_steps = val_gen.samples // val_gen.batch_size
val_gen.reset()
preds = model.predict(val_gen, steps=val_steps, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes[:len(y_pred)]
print('Confusion Matrix:')
print(confusion_matrix(y_true, y_pred))
print('Classification Report:')
print(classification_report(y_true, y_pred, target_names=classes))


# Save final
try:
    model.save(MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to save model: {e}")


