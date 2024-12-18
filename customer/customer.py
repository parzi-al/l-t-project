# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load and preprocess the data
df = pd.read_csv("customer_churn_dataset-training-master.csv")  # Adjust the path
df = df.drop(columns=['CustomerID'])  # Drop unneeded column

# Drop rows with NaN in the target column (Churn)
df = df.dropna(subset=['Churn'])  # Ensure no missing target values

# Encode categorical variables
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

# Split data into features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Check for NaN in features or target and clean if necessary
X = X.dropna()  # Drop rows with NaN in features
y = y[X.index]  # Align y with cleaned X

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for later use
joblib.dump(scaler, "scaler.pkl")

# Define classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=2000),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Naive Bayes': GaussianNB()
}

# Function to evaluate, save, and load models
def train_and_save_model(name, model, X_train, y_train):
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    joblib.dump(model, f"{name.replace(' ', '_')}.pkl")  # Save the model
    print(f"Model {name} saved successfully.\n")

# Train, save, and predict for all models
for name, model in classifiers.items():
    train_and_save_model(name, model, X_train_scaled, y_train)
    print(f"Model {name} training complete!")

# Function to predict using a selected model
def predict_input(model_name, input_data):
    # Load the model and scaler
    model = joblib.load(f"{model_name.replace(' ', '_')}.pkl")
    scaler = joblib.load("scaler.pkl")

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data], columns=X.columns)

    # Encode categorical variables using the same LabelEncoder
    for column in df.select_dtypes(include=['object']).columns:
        input_df[column] = label_encoder.fit_transform(input_df[column])

    # Scale input data
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)
    return prediction

# Example input for prediction
print("\n--- Predicting New Data ---")
new_input = {
    'Age': 30,
    'Gender': 0,  # 0 for Male, 1 for Female
    'Tenure': 39,
    'Usage Frequency': 14,
    'Support Calls': 5,
    'Payment Delay': 18,
    'Subscription Type': 1,  # Encoded value
    'Contract Length': 0,    # Encoded value
    'Total Spend': 932,
    'Last Interaction': 17
}

# Predict churn using a specific model
model_to_use = "Random Forest"
output = predict_input(model_to_use, new_input)
print(f"Prediction using {model_to_use}: {'Churn' if output[0] == 1 else 'No Churn'}")
