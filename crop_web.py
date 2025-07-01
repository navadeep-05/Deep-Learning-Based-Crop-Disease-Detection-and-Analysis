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
                ### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                
                ### About the Plant Disease Recognition System
                This web application aims to assist in the early detection of diseases in crop leaves by leveraging deep learning models. 
                
                ### Purpose
                * This system uses advanced image recognition techniques to classify plant leaves as either healthy or diseased, providing a user-friendly interface where farmers and researchers can upload leaf images for diagnosis. 
                
                * The model used here is based on TensorFlow and has been trained to accurately identify a variety of diseases across different types of crops.
                
                ### Content
                The images have been split into training, validation, and test sets using an 80/20 distribution while maintaining the original directory structure of the dataset:
                
                - train (70295 images)
                - test (33 images)
                - validation (17572 images)

                ### Model and Prediction
                The deep learning model employed here is a convolutional neural network (CNN) built using TensorFlow and Keras. The model has been trained to accurately predict the presence of diseases based on leaf images provided by the user.
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
