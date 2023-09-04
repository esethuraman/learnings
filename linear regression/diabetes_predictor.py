import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def read_data(X_train, y_train):
    X_train_df = pd.DataFrame(X_train)
    y_train_df = pd.DataFrame(y_train)

    print("X_train: ")
    print(X_train_df.head())

    print("y_train")
    print(y_train_df.head())

def evaluate_model(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Evaluation results...")
    print("MSE: ", mse)
    print("R2: ", r2)

#  Load data
# Dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Split into test and train data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, train_size=0.7)

# Load the model.
model = LinearRegression()

# Train the model using training data.
model.fit(X_train, y_train)

# Predict using test data.
y_pred = model.predict(X_test)
print("Predictions: ", y_pred )

evaluate_model(y_test, y_pred)












