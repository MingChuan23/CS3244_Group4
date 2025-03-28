import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and combine data from both .csv
train_data = pd.read_csv('../../Data/fashion-mnist_train.csv')
test_data = pd.read_csv('../../Data/fashion-mnist_test.csv')
all_data = pd.concat([train_data, test_data]).reset_index(drop=True)
x = all_data.drop('label', axis=1)
y = all_data['label']
print(f"{len(x)} input entries with {len(x.columns)} features")
print(f"{len(y)} labels")

# Tried adding 1 augmented image for each image for training (see DT.ipynb)
# However, that worsened the model's performance
# Initially trained a single decision tree but test accuracy was not that great
# Random Forest significantly improved test accuracy
# Hyperparameter tuning using grid search was done for Random Forest algorithm, namely: 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'
# Ideal hyperparameters found to be {'max_depth': 30, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt'}

# Obtain train-test split using test size of 0.2
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

forest = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)

forest.fit(X_train, y_train)

predictions_train = forest.predict(X_train)
predictions_test = forest.predict(X_test)
train_acc = accuracy_score(y_train, predictions_train)
test_acc = accuracy_score(y_test, predictions_test)
print(f"Train Accuracy: {train_acc}")
print(f"Test Accuracy: {test_acc}")
