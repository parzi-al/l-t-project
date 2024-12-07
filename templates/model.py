import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")
print(df.head())
print(df.columns)

df['LotFrontage'].fillna(df['LotFrontage'].median(), inplace=True)
df['MasVnrArea'].fillna(df['MasVnrArea'].median(), inplace=True)

categorical_columns = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
for col in categorical_columns:
    df[col].fillna('None', inplace=True)

print(df.isnull().sum().sum())

df = pd.get_dummies(df, drop_first=True)
features = df[['LotArea', 'OverallQual', 'OverallCond', 'GrLivArea', 'BedroomAbvGr', 'YearBuilt']]
target = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression MSE: {mse_lr}")
print(f"Linear Regression R^2: {r2_lr}")

# Random Forest Regressor model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest Regression MSE: {mse_rf}")
print(f"Random Forest Regression R^2: {r2_rf}")

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("train.csv")
X = df[['LotArea', 'OverallQual', 'GrLivArea', 'BedroomAbvGr', 'YearBuilt', 'TotalBsmtSF', 'GarageArea']]
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

predicted_prices = rf_model.predict(X_test)
threshold_price = 200000  

y_test_categories = ['Luxury' if price > threshold_price else 'Affordable' for price in y_test]
predicted_categories = ['Luxury' if price > threshold_price else 'Affordable' for price in predicted_prices]

def categorize_price(price):
    return 'Luxury' if price > threshold_price else 'Affordable'

predicted_categories = [categorize_price(price) for price in predicted_prices]

results_df = pd.DataFrame({'Predicted Price': predicted_prices, 'Category': predicted_categories})
print(results_df.head())

y_test_categories = [str(label) for label in y_test_categories]
predicted_categories = [str(category) for category in predicted_categories]

accuracy = accuracy_score(y_test_categories, predicted_categories)
print(f'Accuracy: {accuracy:.2f}')

print("Classification Report:")
print(classification_report(y_test_categories, predicted_categories))

print("Confusion Matrix:")
print(confusion_matrix(y_test_categories, predicted_categories))
