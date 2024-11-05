# Deep Learning Based Crop Disease Detection and Analysis

Crop Disease Detection and Analysis Using Deep Learning Models. This project aims to make use of deep learning techniques for accurate detection and classification of crop diseases from leaf images by utilizing advanced neural networks like Convolutional Neural Networks (CNNs) and models like VGG16 and ResNet.

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Working of the Project](#working-of-the-project)
4. [Development Tools and Platforms](#development-tools-and-platforms)
5. [Technologies Used](#technologies-used)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Dataset](#dataset)
9. [Model Architecture](#model-architecture)
10. [Results](#results)
11. [Contributing](#contributing)
12. [License](#license)

## 1. Introduction
Accurate and timely identification of crop diseases is essential for effective crop management and improved yields. This project uses a deep learning model trained on an extensive dataset of crop images to detect diseases, helping farmers and agricultural specialists make informed decisions.

## 2. Features
- **Image-based Crop Disease Detection:** Classifies images of crops into categories such as "healthy" or the specific disease affecting them.
- **User-Friendly Interface:** Built with Streamlit for an interactive experience.
- **Analysis and Insights:** Provides detailed insights into the model's predictions.

## 3. Working of the Project
- **Data Collection and Preprocessing:** The model uses a large dataset of labeled crop images with different diseases. Images are preprocessed by resizing, normalizing, and augmenting to improve model robustness.
- **Model Training:** A Convolutional Neural Network (CNN) model is trained using TensorFlow on the preprocessed images to classify the crop diseases.
- **Model Deployment:** The trained model is integrated into a Streamlit web application, allowing users to upload images and get disease predictions.
- **Prediction and Analysis:** Upon uploading an image, the application processes it through the CNN model and returns a predicted disease label along with confidence scores.

## 4. Development Tools and Platforms
- **GitHub Codespaces:** Used for streamlined development and cloud-based coding.
- **Visual Studio Code:** An alternative editor with built-in support for Git and Python.
- **Streamlit Cloud:** Used for deploying the Streamlit application online.
- **TensorFlow:** Framework for model development and training.

## 5. Technologies Used
- **TensorFlow** - For building and training the CNN model.
- **Streamlit** - For creating a web-based interface.
- **GitHub Codespaces** - For seamless development and deployment.

## 6. Installation
### Prerequisites
Ensure you have:

- Python 3.8 or higher
- git

### Step 1: Clone the Repository
```bash
git clone https://github.com/navadeep-05/Deep-Learning-Based-Crop-Disease-Detection-and-Analysis.git
```
### Step 2: Navigate to the Project Directory
```bash
cd Deep-Learning-Based-Crop-Disease-Detection-and-Analysis
```
### Step 3: Install Required Packages
All required dependencies are listed in the requirements.txt file. Run the following command to install them:
```bash
pip install -r requirements.txt
```

## 7. Usage
1. Run the application:
   ```bash
   streamlit run app.py
   ```
2.  Upload an image of a crop leaf to check for any diseases.
3.  View the prediction and analysis displayed by the application. 

## 8. Dataset
The dataset includes labeled images of various crops with different disease classes, curated from open agricultural sources.

### Download the Datasets
To run this project, download the required datasets from the links below:
- **Training Set**: [Download from Google Drive](https://drive.google.com/drive/folders/1K1KL95qBDhhEAcRzGo74HPyujsCjROpa?usp=drive_link)
- **Testing Set**: [Download from Google Drive](https://drive.google.com/drive/folders/1CUGIR4xMXzp4Gbz1eTgyAkwGV-EAXZRJ?usp=drive_link)
- **Validation Set**: [Download from Google Drive](https://drive.google.com/drive/folders/17EBGIJISjas67T8jMPqspItl0m6wT-sP?usp=drive_link)

### Setup Instructions
1. **Download the Datasets:** Click the links above to access and download the `train`, `test`, and `valid` folders.
2. **Organize the Folders**: After downloading, structure the dataset folders as follows within the repository:
Deep-Learning-Based-Crop-Disease-Detection-and-Analysis/ ├── train/ ├── 
test/ └── valid/
3. **Run the Project**: With the datasets in place, you can run the project in Streamlit:
```bash
streamlit run app.py
 ```

## 9. Model Architecture
A Convolutional Neural Network (CNN) architecture was implemented in TensorFlow. The network consists of multiple convolutional layers, pooling layers, and dense layers to achieve high accuracy in disease detection.

## 10. Results
The model achieved a high accuracy on test data, demonstrating strong performance in identifying diseases across multiple crop types. Example images and their predicted labels are shown within the application.

## 11. Contributing
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

## 12. License
This project is licensed under the MIT License.
