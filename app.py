import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model only once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fashion_mnist_cnn.h5")

model = load_model()

# Class names for Fashion MNIST
class_names = [
    "T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]

st.title("👗 Fashion Item Classifier (Deep Learning Model)")
st.write("Upload an image of a clothing item for prediction.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Load and preprocess the image
    img = Image.open(uploaded_file).convert("L")  # grayscale
    img = img.resize((28, 28))  # resize to model input size

    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(img_array)
    label = class_names[np.argmax(prediction)]

    # Display
    st.image(uploaded_file, caption="Uploaded Image", width=150)
    st.write(f"### Predicted Item: **{label}**")
