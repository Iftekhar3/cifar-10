# To Supress Warnings for streamlit
import warnings
import os
import logging
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO
# Load model
model = tf.keras.models.load_model("my_model.keras")

# CIFAR-10 class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

st.title("🧠 CIFAR-10 Image Classifier")
st.write("Upload an image OR paste an image URL to classify it.")

# Option to choose input type
input_type = st.radio("Choose input type:", ["Upload Image", "Image URL"])

image = None

if input_type == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

elif input_type == "Image URL":
    image_url = st.text_input("Paste an image URL:")
    if image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            st.error("Failed to load image from URL.")

# Proceed if image is available
if image:
    st.image(image, caption="Input Image", use_column_width=True)

    # Preprocess
    image_resized = image.resize((32, 32))
    img_array = np.array(image_resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 32, 32, 3)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    st.markdown("### 🏷️ Prediction:")
    st.success(f"**{predicted_class}**")

    st.markdown("### 📊 Class Probabilities:")
    for i, prob in enumerate(predictions[0]):
        st.write(f"{class_names[i]}: {prob:.4f}")


# https://www.livescience.com/50714-horse-facts.html  - came up as deer