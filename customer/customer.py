import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import h5py
import pickle
import numpy as np

# Load the dataset
df = pd.read_csv('customer_churn_dataset-training-master.csv')

# Handle missing values
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Fill missing values
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
for column in categorical_cols:
    df[column] = df[column].fillna(df[column].mode()[0])

# Encode categorical variables
label_encoder = LabelEncoder()
for column in categorical_cols:
    df[column] = label_encoder.fit_transform(df[column])

# Split data into features and target
X = df.drop(['Churn'], axis=1)
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=2000),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Naive Bayes': GaussianNB()
}

# Dictionary to store trained models
trained_models = {}

# Function to train and evaluate a single model
def evaluate_model(name, model):
    print(f"\n{name}")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Save the trained model in a dictionary
    trained_models[name] = model

    # Performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

# Evaluate and run each model separately
for model_name, model_instance in classifiers.items():
    evaluate_model(model_name, model_instance)

# Save all trained models into an HDF5 file
with h5py.File('customer_models.h5', 'w') as h5f:
    for model_name, model in trained_models.items():
        # Serialize the model using pickle and save as binary in the HDF5 file
        model_bytes = pickle.dumps(model)
        h5f.create_dataset(model_name, data=np.void(model_bytes))

print("\nAll models have been saved in 'customer_models.h5'.")
