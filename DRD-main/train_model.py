import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet import preprocess_input
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os
import sys

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Fix for Unicode issues in some environments
sys.stdout.reconfigure(encoding='utf-8')

# === Paths ===
data_dir = '/Users/pradeepkd/Desktop/projectphase1/archive/colored_images'
model_save_path = "/Users/pradeepkd/Desktop/projectphase1/archive/my_saved_model.h5"

# Ensure the save directory exists
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# === Parameters ===
img_height, img_width = 224, 224
batch_size = 32
epochs = 10
num_classes = 5  # DR has 5 classes

# === Image Generators ===
train_datagen = ImageDataGenerator(
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=preprocess_input  # ResNet preprocessing
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# === Compute class weights ===
classes = list(train_generator.class_indices.keys())
labels = train_generator.classes
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights_dict = dict(enumerate(class_weights))

# Print class distribution and weights for debugging
from collections import Counter
print('Class distribution (training):', Counter(labels))
print('Class indices:', train_generator.class_indices)
print('Class weights:', class_weights_dict)

# === Model Architecture ===
model = Sequential([
    tf.keras.Input(shape=(img_height, img_width, 3)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(512, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# === Compile ===
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === Train ===
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,  # More epochs for custom model
)

# === Evaluate and print confusion matrix ===
val_steps = validation_generator.samples // validation_generator.batch_size
validation_generator.reset()
preds = model.predict(validation_generator, steps=val_steps, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = validation_generator.classes[:len(y_pred)]

# Map folder-based indices to diagnosis numbers
diagnosis_mapping = {'Mild': 1, 'Moderate': 2, 'No_DR': 0, 'Proliferate_DR': 4, 'Severe': 3}
y_pred_diagnosis = np.array([diagnosis_mapping[classes[i]] for i in y_pred])
y_true_diagnosis = np.array([diagnosis_mapping[classes[i]] for i in y_true])

# Diagnosis labels for display
diagnosis_labels = ['0_No_DR', '1_Mild', '2_Moderate', '3_Severe', '4_Proliferate_DR']

print('Confusion Matrix (Diagnosis):')
print(confusion_matrix(y_true_diagnosis, y_pred_diagnosis))
print('Classification Report (Diagnosis):')
print(classification_report(y_true_diagnosis, y_pred_diagnosis, target_names=diagnosis_labels))

# === Save Model (.h5 format) ===
try:
    model.save(model_save_path)
    print(f"✅ Model saved to {model_save_path}")
except Exception as e:
    print(f"❌ Failed to save model: {e}")

# === Optional: Save as TensorFlow SavedModel directory ===
# saved_model_dir = "/Users/pradeepkd/Desktop/projectphase1/archive/my_saved_model"
# try:
#     model.save(saved_model_dir)  # no .h5 extension
#     print(f"✅ SavedModel format saved to {saved_model_dir}")
# except Exception as e:
#     print(f"❌ Failed to save as SavedModel: {e}")
