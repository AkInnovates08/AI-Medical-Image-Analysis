import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Load model
model = load_model("models/best_model.h5")  # ✅ use best model

# Page config
st.set_page_config(page_title="Medical AI", layout="centered")

# Title
st.title("🧠 AI Medical Image Analysis")
st.subheader("Pneumonia Detection from Chest X-ray")

# Upload file
uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

# Prediction function (clean + reusable)
def preprocess_image(image):
    # 🔥 Convert to RGB (fix grayscale issue)
    image = image.convert("RGB")

    # Convert to numpy
    img = np.array(image)

    # Resize
    img = cv2.resize(img, (224, 224))

    # Normalize
    img = img / 255.0

    # Expand dimensions (better than reshape)
    img = np.expand_dims(img, axis=0)

    return img

# If user uploads image
if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img = preprocess_image(image)

    prediction = model.predict(img)[0][0]
    confidence = float(prediction)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded X-ray", width=250)

    with col2:
        st.subheader("🔍 Prediction Result")

        if confidence > 0.5:
            st.error("🛑 Pneumonia Detected")
        else:
            st.success("✅ Normal")

        st.write(f"Confidence: {confidence:.2f}")
        st.progress(min(max(confidence, 0.0), 1.0))

    # Extra info (nice UI touch)
    st.info("⚠️ This is an AI-assisted prediction. Not a medical diagnosis.")