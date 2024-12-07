from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load the dataset
data = pd.read_csv("emails.csv")  # Replace with your dataset file path

# Ensure all entries in the 'text' column are strings
data['text'] = data['text'].astype(str)

# Remove rows with empty strings or whitespace-only strings
data['text'] = data['text'].str.strip()  # Remove leading/trailing whitespace
data = data[data['text'] != '']  # Remove empty rows

# Vectorizing the text using TF-IDF
tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)
X = tfidf.fit_transform(data["text"]).toarray()

# Encode labels (Convert "spam" to numerical if not already)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["spam"])  # Replace "spam" with your label column name

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Keras Sequential model
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")  # Binary classification output
])

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Save the trained model as spam.h5
model.save("spam.h5")
print("Model saved as 'spam.h5'")
