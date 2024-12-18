# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import os
import joblib
# Load the dataset
data = pd.read_csv("emails.csv")  # Replace with the path to your dataset file

# Print the column names to verify them
print("Columns in dataset:", data.columns)

# Assume the email content is stored in a column named 'text' and label as 'spam'
data['text'] = data['text'].astype(str)  # Ensure all email text content are strings
data['spam'] = data['spam'].astype(int)  # Ensure labels are integers (spam = 1, not spam = 0)

# Preprocess data: remove leading/trailing whitespace
data['text'] = data['text'].str.strip()

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(data['text'])  # Features are 'text' column
y = data['spam']  # Target labels (spam = 1, not spam = 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and evaluate using Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
print("\n--- Naive Bayes ---")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# Train and evaluate using Logistic Regression
lr_model = LogisticRegression(max_iter=500, random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# Take input for a new email text and predict if it's spam or not
def predict_spam(email_text):
    email_tfidf = vectorizer.transform([email_text])
    prediction_nb = nb_model.predict(email_tfidf)
    prediction_lr = lr_model.predict(email_tfidf)
    
    print("\nPredictions for given email text:")
    print("Naive Bayes Prediction:", "Spam" if prediction_nb[0] == 1 else "Not Spam")
    print("Logistic Regression Prediction:", "Spam" if prediction_lr[0] == 1 else "Not Spam")

# Example: Input an email text
new_email = input("Enter the email text to check: ")
predict_spam(new_email)


# Ensure the `models/` folder exists
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models")
os.makedirs(model_dir, exist_ok=True)

# Save the models with 'spam' as prefix
print("Saving models...")
joblib.dump(nb_model, os.path.join(model_dir, "spam_naive_bayes.pkl"))  # Naive Bayes Model
joblib.dump(lr_model, os.path.join(model_dir, "spam_logistic_regression.pkl"))  # Logistic Regression Model
# Save the vectorizer
joblib.dump(vectorizer, os.path.join(model_dir, "spam_vectorizer.pkl"))
print("Vectorizer saved successfully.")

print("Models saved successfully in the `models/` folder.")
