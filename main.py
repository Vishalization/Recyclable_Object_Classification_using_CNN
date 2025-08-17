import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model("best_model.keras")

# Get the expected input size (e.g., 150x150)
input_height, input_width = model.input_shape[1:3]

# Define class names (update if your model uses different classes)
class_names = ['Organic', "Recyclable"]

# Prediction function
def predict_image(img: Image.Image):
    img = img.resize((input_width, input_height))  # Resize to model's expected input
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to 0-1

    predictions = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))
    return predicted_class, confidence

# Streamlit UI
st.title("♻️ Waste Classification")
st.markdown("Upload a photo of waste, and the model will predict its category.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Predicting...")
    label, confidence = predict_image(image)
    st.success(f"**Prediction:** {label.capitalize()}")
    st.info(f"**Confidence:** {confidence * 100:.2f}%")
