# 🐱🐶 Cats vs Dogs Image Classifier

An end-to-end **Deep Learning image classification application** that predicts whether an uploaded image contains a **Cat** or a **Dog**, built using **TensorFlow**, **Transfer Learning (MobileNetV2)**, and **Streamlit**.

This project demonstrates the **complete ML lifecycle** — from data ingestion and preprocessing to model training, evaluation, and deployment as a user-friendly web application.

---

## 🚀 Live Demo

🔗 **Streamlit App:**
https://cats-vs-dogs-app-poojana-ometh.streamlit.app/

---

## 📌 Project Overview

- **Problem Statement:**
  Classify images into two categories — **Cat** or **Dog** — with high accuracy.

- **Solution Approach:**
  Use a **Convolutional Neural Network (CNN)** with **transfer learning** to leverage pretrained visual features and deploy the trained model as an interactive web app.

- **Final Model Performance:**
  ✅ **Test Accuracy:** ~96%
  ✅ Balanced Precision, Recall, and F1-score
  ✅ Strong generalization on unseen images

---

## 🧠 Machine Learning Pipeline

### 1️⃣ Data Collection

- Dataset sourced from **Kaggle**
- Over **1,000 labeled images** of cats and dogs
- Images in JPEG format
- Duplicate images removed

**Dataset Link:**
[https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification](https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification)

---

### 2️⃣ Data Preprocessing

- Image resizing to **224 × 224**
- Pixel normalization (0–1 range)
- Train / Validation split (80 / 20)
- Data augmentation:

  - Random horizontal flip
  - Random rotation
  - Random zoom

---

### 3️⃣ Model Training

Three models were trained and compared:

| Model       | Description                            |
| ----------- | -------------------------------------- |
| Model 1     | Custom CNN (Baseline)                  |
| Model 2     | Deeper CNN                             |
| **Model 3** | **MobileNetV2 (Transfer Learning)** ✅ |

**Why MobileNetV2?**

- Pretrained on ImageNet
- Lightweight & efficient
- Strong feature extraction
- Ideal for deployment

---

### 4️⃣ Model Evaluation

#### 🔹 Metrics Used

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix (Train & Test)

#### 🔹 Final Test Results (Best Model)

- **Accuracy:** ~96%
- **Cat Precision:** ~1.00
- **Dog Precision:** ~0.93
- Minimal overfitting

---

### 5️⃣ Model Serialization

- Saved using the **native `.keras` format**
- Ensures compatibility with **Keras 3+**
- Avoids legacy HDF5 (`.h5`) deserialization issues

---

## 🌐 Deployment

### 🔹 Framework

- **Streamlit** (Community Cloud)

### 🔹 Features

- Image upload (JPG / PNG)
- Real-time prediction
- Probability scores
- Confidence-aware warnings
- Mobile-responsive UI
- Cached model loading for performance

---

## 🖥️ Project Structure

```
cats_dogs_app/
│
├── app.py
├── cats_vs_dogs_mobilenetv2.keras
├── CNN_Dog_&_Cat_Classifier.ipynb
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Local Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Ometh4049/cats-vs-dogs-streamlit.git
cd cats-vs-dogs-streamlit
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Application

```bash
streamlit run app.py
```

---

## 📦 Requirements

```txt
streamlit>=1.30.0
tensorflow-cpu>=2.13.0
numpy
pillow
```

---

## 📱 Mobile & UX Design

- Responsive layout
- Touch-friendly components
- Confidence-based feedback
- Clean visual hierarchy
- Minimal sidebar for small screens

---

## 🛡️ Known Issues & Fixes

### ❗ Keras 3 Compatibility

- Legacy `.h5` models may fail to load
- Fixed by:

  - Rebuilding model using Functional API
  - Saving in `.keras` format

---

## 📈 Future Improvements

- Grad-CAM visual explainability
- Batch image upload
- Dockerized deployment
- TensorFlow Lite conversion
- Authentication & analytics

---

## 👨‍💻 Author

**Ometh**
AI & ML Engineer (Aspiring)

---

## © Copyright

© 2026 **Ometh**. All rights reserved.
This project is intended for **educational and demonstration purposes only**.
