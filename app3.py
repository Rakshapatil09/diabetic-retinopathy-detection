import streamlit as st
from PIL import Image
import numpy as np
import os
from tensorflow.keras.models import load_model

# === Load model ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "archive", "my_saved_model.h5")
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    st.success("Model loaded successfully")
else:
    st.error("Model file not found")
    st.stop()

# === Class labels ===
# Alphabetical: Mild, Moderate, No_DR, Proliferate_DR, Severe
labels = {
    0: 'Mild',
    1: 'Moderate',
    2: 'No_DR',
    3: 'Proliferate_DR',
    4: 'Severe'
}

# Map to diagnosis for display (without numbers)
diagnosis_labels = {
    0: 'Mild',
    1: 'Moderate',
    2: 'No_DR',
    3: 'Proliferate_DR',
    4: 'Severe'
}

# === Preprocessing function ===
def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))  # Match training size
    img_array = np.array(img).astype(np.float32)
    
    # Match training preprocessing: simple normalization
    img_array = img_array / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    return img_array

# === Prediction function ===
def predict_uploaded_image(image_file):
    img_array = preprocess_image(image_file)
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = diagnosis_labels[predicted_index]
    confidence = np.max(prediction) * 100
    # Debug: print predictions
    st.write(f"Predicted index: {predicted_index}")
    st.write(f"Prediction probabilities: {prediction[0]}")
    return predicted_class, confidence

# === Global styles (gradient background, glass card, modern buttons) ===
def inject_global_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(1000px 600px at 10% 10%, #eef2ff 0%, rgba(238,242,255,0) 60%),
                        radial-gradient(1000px 600px at 90% 20%, #ecfeff 0%, rgba(236,254,255,0) 60%),
                        linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        }
        .block-container {
            padding-top: 2rem;
        }
        .app-title {
            text-align: center;
            color: #1f2937;
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin: 0.25rem 0 0.5rem 0;
        }
        .app-subtitle {
            text-align: center;
            color: #4b5563;
            font-size: 1.05rem;
            margin-bottom: 1.25rem;
        }
        .glass-card {
            background: rgba(255,255,255,0.75);
            border: 1px solid rgba(148,163,184,0.25);
            border-radius: 18px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 10px 30px rgba(15,23,42,0.08);
            backdrop-filter: blur(10px);
        }
        .stButton>button {
            background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
            color: #ffffff;
            border: none;
            padding: 0.6rem 1.1rem;
            border-radius: 12px;
            font-weight: 600;
            box-shadow: 0 8px 20px rgba(79,70,229,0.25);
            transition: transform 120ms ease, box-shadow 120ms ease;
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 26px rgba(79,70,229,0.3);
        }
        .stFileUploader label {
            font-weight: 600 !important;
            color: #334155 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# === Streamlit App ===
def main():
    inject_global_styles()
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='app-title'>Diabetic Retinopathy Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='app-subtitle'>Upload a retinal fundus image to predict the stage of Diabetic Retinopathy</div>", unsafe_allow_html=True)

    uploaded_image = st.file_uploader("Upload a retinal fundus image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.markdown("""
        <div style='background-color: white; padding: 10px; border-radius: 10px; display: inline-block;'>
        """, unsafe_allow_html=True)
        st.image(uploaded_image, width=400, caption="Uploaded Image")
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("Predict"):
            prediction, confidence = predict_uploaded_image(uploaded_image)
            st.markdown(f"""
<div style='padding: 1.5rem; border-radius: 12px; background: #e6f7ff; border: 2px solid #91d5ff; text-align: center;'>
    <span style='font-size:2rem; font-weight: bold; color: #1890ff;'>🔎 Predicted Class:</span><br>
    <span style='font-size:2.5rem; font-weight: bold; color: #52c41a;'>{prediction}</span>
</div>
""", unsafe_allow_html=True)
            st.markdown(f"""
<div style='padding: 1rem; border-radius: 10px; background: #fffbe6; border: 2px solid #ffe58f; text-align: center; margin-top: 1rem;'>
    <span style='font-size:1.5rem; font-weight: bold; color: #faad14;'>Confidence: {confidence:.2f}</span>
</div>
""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
