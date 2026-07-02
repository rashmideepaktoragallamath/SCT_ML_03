import streamlit as st
import joblib
import cv2
import numpy as np

from skimage.feature import hog
from PIL import Image


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐶",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------

model = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")

IMG_SIZE = 64

st.sidebar.title("📋 Project Information")

st.sidebar.markdown("""
### Model
- Support Vector Machine (SVM)

### Feature Extraction
- Histogram of Oriented Gradients (HOG)

### Dataset
- Kaggle Dogs vs Cats

### Accuracy
- 73.2%

### Kernel
- RBF

### Developer
- Rashmi Deepak Toragallamath
""")

# -----------------------------
# Title
# -----------------------------

st.title("🐶🐱 Cat vs Dog Image Classifier")

st.write("Support Vector Machine (SVM) with HOG Features")

# -----------------------------
# Upload Image
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width=300)

    image = np.array(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    features = scaler.transform([features])

    prediction = model.predict(features)[0]

    label = "🐱 Cat" if prediction == 0 else "🐶 Dog"

    st.success(f"Prediction: {label}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Accuracy", "73.2%")

    with col2:
        st.metric("Algorithm", "SVM")

    with col3:
        st.metric("Feature", "HOG")
st.header("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.image("results/confusion_matrix.png", caption="Confusion Matrix")

with col2:
    st.image("results/sample_predictions.png", caption="Sample Predictions")


    st.markdown("---")

st.header("📖 About")

st.write("""
This application classifies images of cats and dogs using:

- Histogram of Oriented Gradients (HOG)
- Support Vector Machine (SVM)
- StandardScaler
- GridSearchCV for hyperparameter tuning

Dataset: Kaggle Dogs vs Cats
""")