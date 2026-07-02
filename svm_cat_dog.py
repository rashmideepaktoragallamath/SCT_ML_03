import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
import json


from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc

# Dataset Paths
TRAIN_PATH = "dataset/train"
TEST_PATH = "dataset/test"

# Resize all images
IMG_SIZE = 64

def load_images(folder_path):
    images = []
    labels = []
    original_images = []

    categories = ["cats", "dogs"]

    for label, category in enumerate(categories):

        category_path = os.path.join(folder_path, category)

        for image_name in tqdm(os.listdir(category_path), desc=f"Loading {category}"):

            image_path = os.path.join(category_path, image_name)

            image = cv2.imread(image_path)

            if image is None:
                continue

            # Resize image
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            original_images.append(image.copy())

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Extract HOG Features
            features = hog(
                gray,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys"
            )

            images.append(features)
            labels.append(label)
            

    return np.array(images), np.array(labels), original_images


X_train, y_train, train_images = load_images(TRAIN_PATH)
X_test, y_test, test_images = load_images(TEST_PATH)

# Use a subset for faster training
# Shuffle the data
X_train, y_train, train_images = shuffle(
    X_train,
    y_train,
    train_images,
    random_state=42
)

X_test, y_test, test_images = shuffle(
    X_test,
    y_test,
    test_images,
    random_state=42
)

# Use a subset for faster training
X_train = X_train[:2000]
y_train = y_train[:2000]
train_images = train_images[:2000]

X_test = X_test[:500]
y_test = y_test[:500]
test_images = test_images[:500]

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Unique training labels:", np.unique(y_train))
print("Unique testing labels:", np.unique(y_test))
print("Training Images:", X_train.shape)
print("Training Labels:", y_train.shape)

print("Testing Images:", X_test.shape)
print("Testing Labels:", y_test.shape)

# Create the SVM model
print("Searching for the best SVM parameters...")

# Start timer
start_time = time.time()

param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"]
}
 
 

grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

svm_model = grid_search.best_estimator_

# Stop timer
end_time = time.time()

print("Best Parameters:", grid_search.best_params_)
print("Training completed!")



print(f"Training Time: {end_time - start_time:.2f} seconds")
os.makedirs("models", exist_ok=True)

joblib.dump(svm_model, "models/svm_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model saved successfully.")

print("Making predictions...")

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
report = classification_report(
    y_test,
    y_pred,
    target_names=["Cat", "Dog"]
)

print(report)

with open("results/classification_report.txt", "w") as file:
    file.write(report)

# Generate Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Cat", "Dog"],
    yticklabels=["Cat", "Dog"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.savefig("results/confusion_matrix.png")
plt.close()

print("Confusion Matrix saved successfully.")

metrics = {
    "accuracy": round(accuracy * 100, 2),
    "best_parameters": grid_search.best_params_,
    "training_time_seconds": round(end_time - start_time, 2)
}

with open("results/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Metrics saved successfully.")

# ------------------------------------
# Save Sample Predictions
# ------------------------------------

labels = ["Cat", "Dog"]

plt.figure(figsize=(12, 8))

for i in range(6):

    plt.subplot(2, 3, i + 1)

    image = cv2.cvtColor(test_images[i], cv2.COLOR_BGR2RGB)

    plt.imshow(image)

    actual = labels[y_test[i]]
    predicted = labels[y_pred[i]]

    color = "green" if actual == predicted else "red"

    plt.title(
        f"Actual: {actual}\nPredicted: {predicted}",
        color=color,
        fontsize=10
    )

    plt.axis("off")

plt.tight_layout()

plt.savefig("results/sample_predictions.png")

plt.close()

print("Sample predictions saved successfully.")

# ------------------------------------
# ROC Curve
# ------------------------------------

y_scores = svm_model.decision_function(X_test)

fpr, tpr, _ = roc_curve(y_test, y_scores)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()

plt.savefig("results/roc_curve.png")

plt.close()

print("ROC Curve saved successfully.")