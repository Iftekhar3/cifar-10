# CIFAR-10 Image Classification with Streamlit Web App

An end-to-end deep learning project for classifying images into 10 categories from the CIFAR-10 dataset.  
The project covers the full ML lifecycle — from data preprocessing and model training to deployment as an interactive Streamlit web application.

## 🚀 Project Overview
This project demonstrates how to:
- Build and train a custom **Convolutional Neural Network (CNN)** for image classification.
- Use **data augmentation** and **training callbacks** to improve performance and prevent overfitting.
- Save and reload trained models for future retraining or deployment.
- Deploy a **Streamlit web app** where users can upload images and instantly receive classification predictions.

---

## 📊 Dataset
- **Name:** CIFAR-10  
- **Source:** [Kaggle CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)  
- **Description:** 60,000 color images (32x32 pixels) across 10 classes:
  - Airplane
  - Automobile
  - Bird
  - Cat
  - Deer
  - Dog
  - Frog
  - Horse
  - Ship
  - Truck

---

## 🛠 Tech Stack
- **Languages & Libraries:** Python, NumPy, Matplotlib
- **ML/DL Frameworks:** TensorFlow, Keras
- **Deployment:** Streamlit
- **Version Control:** Git, GitHub

---

## ⚙️ Features & Implementation
- **Model Architecture:**
  - Multiple Conv2D + BatchNorm layers
  - MaxPooling & Dropout for regularization
  - Dense output layer with softmax activation
- **Training Optimizations:**
  - Data augmentation (rotation, flipping, shifting)
  - `ReduceLROnPlateau` to adjust learning rate dynamically
  - `EarlyStopping` to avoid overfitting
- **Deployment:**
  - Streamlit interface for real-time image classification
  - Local deployment on `localhost`

---

## 📈 Results
- **Training Accuracy:** ~93%
- **Validation Accuracy:** ~92%
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam (lr = 0.001)

---

## 🖥 Streamlit Web App
The Streamlit app allows users to:
1. Upload an image (JPG/PNG).
2. See real-time predictions with confidence scores.
3. Test model performance interactively.

**Run the app locally:**
```bash streamlit run app.py
