# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf

# Load model once globally
model = load_model('plant_disease_model.h5')

# Class labels
CLASS_NAMES = ['Tomato-Bacterial_spot', 'Potato-Barly blight', 'Corn-Common_rust']

# Sidebar navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    st.image("crop home page.jpg", use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases.

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** State-of-the-art machine learning techniques for accurate detection.
    - **User-Friendly:** Simple and intuitive interface.
    - **Fast and Efficient:** Results in seconds for quick action.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to begin!
    
    ### About Us
    Learn more on the **About** page.
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset contains ~87K RGB images of healthy and diseased crop leaves across 38 classes. It is split into training, validation, and test sets.

    ### Project Purpose
    Assist in early disease detection using deep learning for improved crop yield.

    ### System
    - Built with TensorFlow/Keras
    - Uses CNN model to classify diseases in crop leaves

    ### Dataset Split
    - Train: 70295 images
    - Validation: 17572 images
    - Test: 33 images
    """)

# Disease Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    st.markdown("Upload a plant leaf image to detect disease")

    plant_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if st.button("Show Image"):
        if plant_image is not None:
            file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            st.image(opencv_image, caption="Uploaded Image", channels="BGR", use_column_width=True)

    if st.button("Predict Disease"):
        if plant_image is not None:
            file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            opencv_image = cv2.resize(opencv_image, (256, 256))
            opencv_image.shape = (1, 256, 256, 3)

            Y_pred = model.predict(opencv_image)
            result = CLASS_NAMES[np.argmax(Y_pred)]

            st.success("Model is predicting:")
            st.title(str("This is a " + result.split('-')[0] + " leaf with " + result.split('-')[1]))

            # Visual effects
            st.snow()
            st.balloons()
            st.toast("Prediction Complete!", icon="🌿")
