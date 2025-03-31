import pandas as pd
import numpy as np
# import cv2
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    top_k_accuracy_score,
    roc_auc_score,
    log_loss
)
from sklearn.preprocessing import label_binarize
from memory_profiler import memory_usage
print("import done")

# Load dataset
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

# def extract_sift_features(images):
#     sift = cv2.SIFT_create()
#     all_features = []

#     for img in images:
#         kp, des = sift.detectAndCompute(img, None)
#         if des is not None:
#             # Take mean of all descriptors in this image
#             feature = np.mean(des, axis=0)
#         else:
#             # No keypoints found; use zeros
#             feature = np.zeros(128)
#         all_features.append(feature)

#     return np.array(all_features)


# # Feature extraction using SIFT
# X_raw_full = df_train.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)
# X_raw_subset = X_raw_full[:60000]  
# X_sift_subset = extract_sift_features(X_raw_subset)
# y_full = df_train.iloc[:, 0].values
# y_subset = y_full[:60000]
# print("60000 used")


# # Use important features to train ie (60000,128) instead of (60000,784)
# X_train_sift, X_val_sift, y_train_sift, y_val_sift = train_test_split(X_sift_subset, y_subset, test_size=0.2, random_state=42)
# print("splitting done for validation set")

# svm_model = SVC(kernel='rbf')  # rbf/poly better for image, linear is for text
# svm_model.fit(X_train_sift, y_train_sift)

# y_pred = svm_model.predict(X_val_sift)
# print("Validation - training set split 48000 vs 12000")
# print("Accuracy:", accuracy_score(y_val_sift, y_pred))
# print(classification_report(y_val_sift, y_pred))

# # we now test on unseen data df_test

# X_test_raw = df_test.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)
# y_test = df_test.iloc[:, 0].values

# X_test_sift = extract_sift_features(X_test_raw)
# print("\nSifting done on unseen data")

# y_test_pred = svm_model.predict(X_test_sift)

# print("\n=== Final Evaluation on df_test (Unseen Data) ===")
# print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
# print("Test Error:", 1 - accuracy_score(y_test, y_test_pred))
# print("Classification Report:\n", classification_report(y_test, y_test_pred))

########################################## use full 784 pixels

# Extract features and labels
X_train_raw = df_train.iloc[:, 1:].values # 60000 x 784
y_train = df_train.iloc[:, 0].values # 60000 x 1

X_test_raw = df_test.iloc[:, 1:].values
y_test = df_test.iloc[:, 0].values

# Normalize
X_train = X_train_raw / 255.0
X_test = X_test_raw / 255.0
print("normalisation done")

# Train SVM (with probability enabled)
svm_model = SVC(kernel='rbf', probability = True)
svm_model.fit(X_train, y_train)
y_train_pred = svm_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print("Finished training")

# --- Inference Time ---
start_time = time.time()
y_pred = svm_model.predict(X_test)
end_time = time.time()
inference_time = end_time - start_time
print(f"\nInference Time: {inference_time:.4f} seconds")

# # --- Memory Usage ---
# def predict_with_memory():
#     return svm_model.predict(X_test)

# mem_usage = memory_usage(predict_with_memory)
# print(f"Memory Usage during inference: {max(mem_usage) - min(mem_usage):.2f} MiB")

# --- Accuracy & Report ---
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Error: {1 - test_accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Class Accuracy ---
print("\nClass-wise Accuracy:")
for label in np.unique(y_test):
    idx = (y_test == label)
    acc = accuracy_score(y_test[idx], y_pred[idx])
    print(f"Class {label} ({fashion_mnist_labels[label]}): {acc:.2f}")

# --- Top-k Accuracy ---
y_pred_proba = svm_model.predict_proba(X_test)
top3_acc = top_k_accuracy_score(y_test, y_pred_proba, k=3)
print(f"\nTop-3 Accuracy: {top3_acc:.2f}")

# --- AUC-ROC ---
y_test_bin = label_binarize(y_test, classes=np.arange(10))
roc_auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr')
print(f"AUC-ROC Score (OvR): {roc_auc:.2f}")

# --- Log Loss ---
logloss = log_loss(y_test, y_pred_proba)
print(f"Log Loss: {logloss:.4f}")

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=fashion_mnist_labels.values(),
            yticklabels=fashion_mnist_labels.values(), cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()