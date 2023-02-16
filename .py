import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=32)

# Define KNN classifier
knn = KNeighborsClassifier()

# Define hyperparameter search space
params = {'n_neighbors': list(range(1, 12)), 'p': [1, 2, 3, 4]}

# Perform random search for optimal hyperparameters
random_search = RandomizedSearchCV(knn, params, cv=3, n_iter=6, n_jobs=-2)
random_search.fit(X_train, y_train)
knn = random_search.best_estimator_

# Train KNN classifier
knn.fit(X_train, y_train)

# Evaluate performance using k-fold cross-validation
scores = cross_val_score(knn, X_train, y_train, cv=5)
print(f"Mean accuracy: {np.mean(scores):.2f}, Standard deviation: {np.std(scores):.2f}")

# Evaluate performance on test set
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# Print performance metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Sensitivity: {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
