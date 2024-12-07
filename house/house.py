import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError

# Load data
df = pd.read_csv("train.csv")

# Define features and target
X = df[['LotArea', 'OverallQual', 'OverallCond', 'BedroomAbvGr', 'YearBuilt', 
        'TotalBsmtSF', 'GrLivArea']]
y = df['SalePrice']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a simple neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the model with MeanSquaredError
model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mae'])
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Save the model as a TensorFlow .h5 file
model.save("house.h5")
print("Model saved as models/model.h5")
