import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # For saving/loading models
import os

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "train.csv")
df = pd.read_csv(data_path)

# Fill missing values for numerical columns with the median
df['LotFrontage'].fillna(df['LotFrontage'].median(), inplace=True)
df['MasVnrArea'].fillna(df['MasVnrArea'].median(), inplace=True)

# Fill missing values for categorical columns with 'None'
categorical_columns = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
for col in categorical_columns:
    df[col].fillna('None', inplace=True)

# Features and target variable
features = df[['LotArea', 'OverallQual', 'OverallCond', 'GrLivArea', 'BedroomAbvGr', 'YearBuilt']]
target = df['SalePrice']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train models
print("Training Linear Regression model...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print("Training Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate models (optional)
print("Evaluating models on test data...")
lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

lr_rmse = mean_squared_error(y_test, lr_pred, squared=False)
rf_rmse = mean_squared_error(y_test, rf_pred, squared=False)

lr_r2 = r2_score(y_test, lr_pred)
rf_r2 = r2_score(y_test, rf_pred)

print(f"Linear Regression - RMSE: {lr_rmse:.2f}, R^2: {lr_r2:.2f}")
print(f"Random Forest - RMSE: {rf_rmse:.2f}, R^2: {rf_r2:.2f}")

# Ensure the `model/` folder exists
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models")
os.makedirs(model_dir, exist_ok=True)

# Save the models
print("Saving models...")
joblib.dump(lr_model, os.path.join(model_dir, "house_linear_regression_model.pkl"))
joblib.dump(rf_model, os.path.join(model_dir, "house_random_forest_model.pkl"))

print("Models saved successfully in the `model/` folder.")
