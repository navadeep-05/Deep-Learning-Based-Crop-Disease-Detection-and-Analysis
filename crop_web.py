import streamlit as st
import tensorflow as tf
import numpy as np

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('crop_disease_model.h5')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = r"crop home page.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                ## About the Plant Disease Recognition System
                This project aims to assist in the early detection of diseases in crop leaves by leveraging deep learning models. Early detection of plant diseases can significantly impact agricultural productivity, enabling farmers to take preventive measures before the diseases spread.
                ### Purpose
                * This system uses advanced image recognition techniques to classify plant leaves as either healthy or diseased, providing a user-friendly interface where farmers and researchers can upload leaf images for diagnosis. 
                
                * The model used here is based on TensorFlow and has been trained to accurately identify a variety of diseases across different types of crops.
                #### Content
                The images have been split into training, validation, and test sets using an 80/20 distribution while maintaining the original directory structure of the dataset:
                
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                ### Model and Prediction
                The deep learning model employed here is a convolutional neural network (CNN) built using TensorFlow and Keras. The model has been trained to accurately predict the presence of diseases based on leaf images provided by the user.
                """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")

    def crop_disease_suggestions(disease):
        treatment_suggestions = {
            "Apple___Apple_scab": ["Apply fungicides containing sulfur or copper", 
                                   "improve air circulation around trees by pruning infected branches and leaves.",
                                   "Avoid overhead irrigation to reduce leaf wetness."],
            "Apple___Black_rot": ["Remove and destroy infected fruit and branches",
                                  "Apply fungicides containing myclobutanil, chlorothalonil or captan."],
                 "Apple___Cedar_apple_rust": ["Remove and destroy infected leaves and galls to reduce spore production",
                                              "Spray fungicides containing acylamino acid or sulfur",
                                               "Avoid planting apple trees near cedar trees."
                                               "Use resistant apple cultivars."],
                 "Blueberry___healthy": ["No treatment necessary for healthy blueberry plants but ensure proper care such as mulching, avoiding water stress and fertilizing approximately."],
                 "Cherry_(including_sour)__Powdery_mildew": ["Apply fungicides containing sulfur, potassium bicarbonate, or neem oil."
                                                             "Improve air circulation through pruning.", 
                                                             "Avoid overhead watering."
                                                             "Ensure proper spacing between trees."],
                 "Cherry(including_sour)__healthy": ["No treatment necessary for healthy cherry trees."
                                                     "Continue standard care practices like pruning and irrigation."],
                 "Corn(maize)__Cercospora_leaf_spot Gray_leaf_spot": ["Rotate crops with non-host plants (such as soybeans or small grains).",
                                                                      "Remove and destroy infected plant debris at the end of the season.", 
                                                                      "Apply fungicides containing chlorothalonil or strobilurin at early infection stages."],
                 "Corn(maize)_Common_rust": ["Plant rust-resistant corn varieties.", 
                                             "Avoid planting corn in low-lying and humid areas.", 
                                             "Apply fungicides containing triazole or strobilurin if severe infection occurs."],
                 "Corn(maize)__Northern_Leaf_Blight": ["Use plant resistant varieties.", 
                                                       "Rotate crops to reduce pathogen buildup.", 
                                                       "Apply fungicides containing chlorothalonil or strobilurin."
                                                       "Improve air circulation by planting rows in the direction of the prevailing wind."],
                 "Corn(maize)healthy": ["No treatment necessary for healthy corn plants but Maintain proper crop rotation, soil fertility, and irrigation practices."],
                 "Grape___Black_rot": ["Remove and destroy infected fruit and vines.", 
                                       "Prune infected wood  to improve air circulation.", 
                                       "Apply fungicides containing sulfur or captan at the pre-bloom stage."
                                       "Use disease-resistant grape varieties."],
                 "Grape___Esca(Black_Measles)": ["Remove and destroy infected vines.",
                                                 "Avoid planting grapes in poorly drained areas or waterlogged areas.",  
                                                 "Maintain proper vineyard sanitation."],
                 "Grape___Leaf_blight(Isariopsis_Leaf_Spot)": ["Remove and destroy infected leaves, avoid overhead watering",
                                                                "Apply fungicides containing copper or sulfur."
                                                                "Aoid overhead watering to reduce leaf wetness."],
                 "Grape___healthy": ["No treatment necessary for healthy grapevines but maintain good vineyard practices such as proper irrigation and pest control.."],
                 "Orange___Haunglongbing(Citrus_greening)": ["There is currently no cure for citrus greening but early detection and removal of infected trees can help prevent its spread."
                                                             "Remove and destroy infected trees to prevent further spread."],
                 "Peach___Bacterial_spot": ["Remove and destroy infected twigs and branches.", 
                                            "Avoid overhead irrigation and wetting leaves.", 
                                            "Apply copper-based fungicides."
                                            "Use Plant resistant peach varieties."],
                 "Peach___healthy": ["No treatment necessary for healthy peach trees but Regular care practices such as pruning and watering are sufficient."],
                 "Pepper,_bell___Bacterial_spot": ["Remove and destroy infected plants.", 
                                                   "Avoid overhead watering.", 
                                                   "Apply copper-based fungicides."
                                                   "Rotate crops with non-susceptible plants like corn or beans."],
                 "Pepper,_bell___healthy": ["No treatment necessary for healthy bell pepper plants but continue with regular watering and fertilization practices.."],
                 "Potato___Early_blight": [" Use plant resistant varieties.", 
                                           "Rotate crops to reduce pathogen buildup.", 
                                           "Apply fungicides containing chlorothalonil or mancozeb."
                                           "Remove and destroy infected plant debris."
                                           "Avoid overhead irrigation."],
                 "Potato___Late_blight": ["Use plant resistant varieties.", 
                                          "Rotate crops, apply fungicides containing chlorothalonil or maneb.", 
                                          "Avoid overhead watering."
                                          "Avoid overhead irrigation to reduce leaf wetness."],
                 "Potato___healthy": ["No treatment necessary for healthy potato plants but maintain regular watering and fertilization."],
                 "Raspberry___healthy": ["No treatment necessary for healthy raspberry plants but ensure proper pruning and watering."],
                 "Soybean___healthy": ["No treatment necessary for healthy soybean plants but use good agricultural practices like crop rotation and irrigation."],
                 "Squash___Powdery_mildew": ["Apply fungicides containing sulfur or potassium bicarbonate.", 
                                             "Improve air circulation  by proper spacing of plants.",
                                             "Avoid overhead watering."],
                 "Strawberry___Leaf_scorch": ["Avoid overwatering.", 
                                              "Improve air circulation.", 
                                              "Remove and destroy infected plants."
                                              "Apply fungicides like captan or myclobutanil if necessary."],
                 "Strawberry___healthy": ["No treatment necessary for healthy strawberry plants but continue with regular care practices."],
                 "Tomato___Bacterial_spot": ["Remove and destroy infected plants.", 
                                             "Avoid overhead watering.", 
                                             "Apply copper-based fungicides."],
                 "Tomato___Early_blight": ["Rotate crops.", 
                                           "Use plant resistant varieties.", 
                                           "Apply fungicides containing chlorothalonil or maneb."
                                           "Remove and destroy infected plants."],
                 "Tomato___Late_blight": ["Apply fungicides containing mancozeb or chlorothalonil.", 
                                          "Remove infected plants.",
                                          "Avoid overhead watering.",
                                          "Use resistant varieties."],
                 "Tomato___Leaf_Mold": ["Improve air circulation.", 
                                        "Avoid overhead watering.", 
                                        "Apply fungicides containing sulfur or potassium bicarbonate."],
                 "Tomato___Septoria_leaf_spot": ["Remove and destroy infected plants.", 
                                                 "Avoid overhead watering.", 
                                                 "Apply fungicides containing chlorothalonil or maneb."],
                 "Tomato___Spider_mites Two-spotted_spider_mite": ["Use insecticidal soap, neem oil, or predatory mites to control spider mites."],
                 "Tomato___Target_Spot": ["Remove and destroy infected plants.", 
                                          "Avoid overhead watering.", 
                                          "Apply fungicides containing chlorothalonil or maneb."],
                 "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ["Remove and destroy infected plants.",
                                                            "Clean tools and equipment.",
                                                            "Plant virus-resistant varieties."],
                 "Tomato___Tomato_mosaic_virus": ["Remove and destroy infected plants, clean tools and equipment, and plant virus-resistant varieties."],
                 "Tomato___healthy": ["No treatment necessary for healthy tomato plants."],
            }
        return treatment_suggestions.get(disease, ["No specific treatment suggestions found for this disease."])

    # Upload image
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    if(st.button("Show Image")):
        if test_image is not None:
            # Show the image
            st.image(test_image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("Predict"):
        with st.spinner("Please wait, predicting..."):
            result_index = model_prediction(test_image)

            # Reading Labels
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
                ]

            # Model prediction result
            predicted_disease = class_name[result_index]
            st.success(f"Model is predicting: {predicted_disease}")

            # Show effects
            st.snow()
            st.balloons()
            st.toast("Prediction Complete!", icon="🍃")

            # Show treatment suggestions
            suggestions = crop_disease_suggestions(predicted_disease)
            st.subheader("Treatment Suggestions:")
            for suggestion in suggestions:
                st.write(f"- {suggestion}")

