import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, Rescaling
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from data_split import make_generators


# Fix stdout encoding on some environments
sys.stdout.reconfigure(encoding='utf-8')


# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive/colored_images")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive/my_saved_model_efficientnet.h5")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


# Params
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50


# Data Generators
# Images are scaled to [0, 1] to match app3.py; the model rescales back to the
# [0, 255] range EfficientNet expects (see the Rescaling layer below).
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

# Split per original image so balance_dataset.py copies can't leak into validation
train_gen, val_gen = make_generators(DATA_DIR, train_datagen, val_datagen,
                                     target_size=IMG_SIZE, batch_size=BATCH_SIZE)


# Class weights
classes = list(train_gen.class_indices.keys())
labels = train_gen.classes
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights_dict = dict(enumerate(class_weights))


# Model
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3))
base_model.trainable = False  # first stage: freeze

inputs = Input(shape=(*IMG_SIZE, 3))
x = Rescaling(255.0)(inputs)  # app feeds [0, 1]; EfficientNet expects [0, 255]
x = base_model(x, training=False)  # keep BatchNorm in inference mode while fine-tuning
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
outputs = Dense(train_gen.num_classes, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)

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
val_gen.reset()
preds = model.predict(val_gen, verbose=1)
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


