#!/usr/bin/env python3
"""
Debug script to test model predictions on known images
"""
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from PIL import Image

# Load the model
model_path = "/Users/pradeepkd/Desktop/projectphase1/archive/my_saved_model.h5"
model = load_model(model_path)

# Test on a few images from each class
data_dir = '/Users/pradeepkd/Desktop/projectphase1/archive/colored_images'
classes = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

print("Testing model predictions on known images:")
print("=" * 60)

for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)
    if os.path.exists(class_dir):
        # Get first image from this class
        image_files = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            test_image = os.path.join(class_dir, image_files[0])
            
            # Preprocess and predict
            img_array = preprocess_image(test_image)
            prediction = model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            print(f"True class: {class_name}")
            print(f"Predicted class index: {predicted_class_idx}")
            print(f"Confidence: {confidence:.2f}%")
            print(f"All class probabilities: {prediction[0]}")
            print("-" * 40)

# Also check what the model's class indices should be
print("\nModel summary:")
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")
print(f"Number of classes: {model.output_shape[1] if len(model.output_shape) > 1 else 'Unknown'}")

# Check if there's any way to get class names from the model
try:
    if hasattr(model, 'class_indices'):
        print(f"Model class indices: {model.class_indices}")
    else:
        print("No class_indices found in model")
except:
    print("Could not access class_indices from model")
