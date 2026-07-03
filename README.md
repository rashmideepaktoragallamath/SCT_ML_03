# 🐶🐱 Cat vs Dog Image Classifier using SVM & HOG

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-SVM-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)


A Machine Learning project that classifies **Cats** and **Dogs** using:

- Histogram of Oriented Gradients (HOG)
- Support Vector Machine (SVM)
- Hyperparameter Tuning with GridSearchCV
- Streamlit Web Application

---

# 🌐 Live Demo

**Try the application here**

👉 **PASTE YOUR STREAMLIT LINK HERE**

---

# 📌 Project Overview

This project uses traditional Machine Learning techniques for image classification.

Instead of Deep Learning, it extracts HOG (Histogram of Oriented Gradients) features from images and trains an optimized Support Vector Machine classifier.

The project also includes an interactive Streamlit web application where users can upload images and receive predictions instantly.

---

# ✨ Features

- Image Upload
- Real-time Prediction
- HOG Feature Extraction
- SVM Classifier
- GridSearchCV Hyperparameter Optimization
- StandardScaler
- Confusion Matrix
- Classification Report
- ROC Curve
- Sample Predictions
- Streamlit Web Interface

---

# 🧠 Machine Learning Pipeline

Dataset

↓

Image Preprocessing

↓

Resize Images (64×64)

↓

Convert to Grayscale

↓

Extract HOG Features

↓

Standard Scaling

↓

GridSearchCV

↓

Train SVM

↓

Save Model (.pkl)

↓

Streamlit Deployment

---

# 📊 Model Performance

| Metric | Value |
|---------|--------|
| Accuracy | **73.2%** |
| Algorithm | SVM |
| Kernel | RBF |
| Feature Extraction | HOG |

---

# 📷 Screenshots

## Web Application

Add a screenshot here.

---

## Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

---

## Sample Predictions

![Predictions](results/sample_predictions.png)

---

## ROC Curve

![ROC Curve](results/roc_curve.png)

---

# 📂 Dataset

Dataset Used:

**Dogs vs Cats**

Source:

https://www.kaggle.com/c/dogs-vs-cats

> The dataset is not included in this repository because of its size.

Folder Structure:

dataset/

├── train/

│ ├── cats/

│ └── dogs/

└── test/

├── cats/

└── dogs/

---

# 🛠 Tech Stack

- Python
- OpenCV
- NumPy
- Scikit-learn
- Scikit-image
- Matplotlib
- Joblib
- Streamlit

---

# 📁 Project Structure

```
SCT_ML_03
│
├── app.py
├── svm_cat_dog.py
├── requirements.txt
├── README.md
│
├── models
│   ├── svm_model.pkl
│   └── scaler.pkl
│
├── results
│   ├── confusion_matrix.png
│   ├── sample_predictions.png
│   ├── roc_curve.png
│   └── classification_report.txt
│
└── dataset
```

---

# ⚙ Installation

Clone Repository

```bash
git clone YOUR_REPO_LINK
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit App

```bash
streamlit run app.py
```

---

# 🎯 Future Improvements

- CNN-based Classification
- Confidence Score Display
- Batch Image Prediction
- Explainable AI (Grad-CAM)
- Docker Deployment
- Cloud Deployment using AWS

---

# 👨‍💻 Developer

**Rashmi Deepak Toragallamath**

Machine Learning Enthusiast

GitHub:
PASTE YOUR GITHUB PROFILE

LinkedIn:
PASTE YOUR LINKEDIN PROFILE

---

⭐ If you found this project helpful, consider giving it a Star!