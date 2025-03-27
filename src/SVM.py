import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

df_train = pd.read_csv("../data/fashion-mnist_train.csv")
df_test = pd.read_csv("../data/fashion-mnist_test.csv")

fashion_mnist_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}


def extract_sift_features(images):
    sift = cv2.SIFT_create()
    all_features = []

    for img in images:
        kp, des = sift.detectAndCompute(img, None)
        if des is not None:
            # Take mean of all descriptors in this image
            feature = np.mean(des, axis=0)
        else:
            # No keypoints found; use zeros
            feature = np.zeros(128)
        all_features.append(feature)

    return np.array(all_features)


# Feature extraction using SIFT
X_raw_full = df_train.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)
X_raw_subset = X_raw_full[:5000]  
X_sift_subset = extract_sift_features(X_raw_subset)
y_full = df_train.iloc[:, 0].values
y_subset = y_full[:5000]


# Use important features to train ie (60000,128) instead of (60000,784)
X_train_sift, X_val_sift, y_train_sift, y_val_sift = train_test_split(X_sift_subset, y_subset, test_size=0.2, random_state=42)
print("splitting done")


svm_model = SVC(kernel='linear')  # rbf or poly?
svm_model.fit(X_train_sift, y_train_sift)

y_pred = svm_model.predict(X_val_sift)
print("Accuracy:", accuracy_score(y_val_sift, y_pred))
print(classification_report(y_val_sift, y_pred))
