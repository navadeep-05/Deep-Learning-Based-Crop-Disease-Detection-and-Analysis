# Deep Learning Based Crop Disease Detection and Analysis

Crop Disease Detection and Analysis Using Deep Learning Models. This project aims to make use of deep learning techniques for accurate detection and classification of crop diseases from leaf images by utilizing advanced neural networks like Convolutional Neural Networks (CNNs) and models like VGG16 and ResNet.

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Dataset](#dataset)
7. [Model Architecture](#model-architecture)
8. [Results](#results)
9. [Contributing](#contributing)
10. [License](#license)

## 1. Introduction
Accurate and timely identification of crop diseases is essential for effective crop management and improved yields. This project uses a deep learning model trained on an extensive dataset of crop images to detect diseases, helping farmers and agricultural specialists make informed decisions.

## 2. Features
- **Image-based Crop Disease Detection:** Classifies images of crops into categories such as "healthy" or the specific disease affecting them.
- **User-Friendly Interface:** Built with Streamlit for an interactive experience.
- **Analysis and Insights:** Provides detailed insights into the model's predictions.

## 3. Technologies Used
- **TensorFlow** - For building and training the CNN model.
- **Streamlit** - For creating a web-based interface.
- **GitHub Codespaces** - For seamless development and deployment.

## 4. Installation
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

## 5. Usage
1. Run the application:
   ```bash
   streamlit run app.py
   ```
2.  Upload an image of a crop leaf to check for any diseases.
3.  View the prediction and analysis displayed by the application. 

## 6. Dataset
The dataset includes labeled images of various crops with different disease classes, curated from open agricultural sources.

## 7. Model Architecture
A Convolutional Neural Network (CNN) architecture was implemented in TensorFlow. The network consists of multiple convolutional layers, pooling layers, and dense layers to achieve high accuracy in disease detection.

## 8. Results
The model achieved a high accuracy on test data, demonstrating strong performance in identifying diseases across multiple crop types. Example images and their predicted labels are shown within the application.

## 9. Contributing
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

## 10. License
This project is licensed under the MIT License.
