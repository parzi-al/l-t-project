from flask import Flask, render_template, request, jsonify
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits

app = Flask(__name__)

# Paths
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(static_dir, exist_ok=True)

# Load Models and Check Existence
def load_model(file_name):
    try:
        model_path = os.path.join(model_dir, file_name)
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            raise FileNotFoundError(f"{file_name} not found.")
    except Exception as e:
        print(f"Error loading {file_name}: {str(e)}")
        return None

models = {
    'digits_pca': load_model('digits_pca_model.pkl'),
    'digits_tsne': load_model('digits_tsne_model.pkl'),
    'house_linear': load_model('house_linear_regression_model.pkl'),
    'house_rf': load_model('house_random_forest_model.pkl'),
    'churn_linear': load_model('Logistic_Regression.pkl'),
    'churn_rf': load_model('Random_Forest.pkl'),
    'spam_logistic': load_model('spam_logistic_regression.pkl'),
    'spam_nb': load_model('spam_naive_bayes.pkl'),
    'churn_nb': load_model('Naive_Bayes.pkl'),
    'churn_gb': load_model('Gradient_Boosting.pkl'),
    'churn_knn': load_model('K-Nearest_Neighbors.pkl'),
    'churn_svm': load_model('Support_Vector_Machine.pkl')
}

# Load Scalers
scalers = {
    'churn_scaler': load_model('cust_scaler.pkl')
}

# Load the TF-IDF vectorizer
vectorizer = load_model('spam_vectorizer.pkl')

# Load Digits Dataset
digits = load_digits()

# Visualize sample images
def plot_sample_images(data, labels, n=10):
    plt.figure(figsize=(10, 5))
    for index, (image, label) in enumerate(zip(data[:n], labels[:n])):
        plt.subplot(2, n // 2, index + 1)
        plt.imshow(image.reshape(8, 8), cmap=plt.cm.gray)
        plt.title(f'Label: {label}')
        plt.axis('off')
    plt.savefig(os.path.join(static_dir, "sample_images.png"))
    plt.close()

plot_sample_images(digits.data, digits.target)

def plot_pca(X_pca, y):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="viridis", legend="full", s=60)
    plt.title('PCA: MNIST Digits in 2D')
    plt.savefig(os.path.join(static_dir, "pca_plot.png"))
    plt.close()

def plot_tsne(X_tsne, y):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette="coolwarm", legend="full", s=60)
    plt.title('t-SNE: MNIST Digits in 2D')
    plt.savefig(os.path.join(static_dir, "tsne_plot.png"))
    plt.close()

@app.route("/", methods=["GET", "POST"])
def index():
    predictions = {}
    show_images = False
    error = None

    if request.method == "POST":
        try:
            if "generate_plots" in request.form:
                # Generate plots synchronously
                X_pca = models['digits_pca'].transform(digits.data)
                plot_pca(X_pca, digits.target)
                X_tsne = models['digits_tsne'].fit_transform(digits.data)
                plot_tsne(X_tsne, digits.target)
                show_images = True

            # House Price Predictions
            house_features = [
                float(request.form.get("LotArea", 0)),
                int(request.form.get("OverallQual", 0)),
                int(request.form.get("OverallCond", 0)),
                float(request.form.get("GrLivArea", 0)),
                int(request.form.get("BedroomAbvGr", 0)),
                int(request.form.get("YearBuilt", 0)),
            ]
            house_features = np.array(house_features).reshape(1, -1)
            predictions['house_linear'] = models['house_linear'].predict(house_features)[0]
            predictions['house_rf'] = models['house_rf'].predict(house_features)[0]

            # Customer Churn Prediction
            churn_features = [
                float(request.form.get("Age", 0)),
                1 if request.form.get("Gender", "Male") == "Female" else 0,
                int(request.form.get("Tenure", 0)),
                int(request.form.get("UsageFrequency", 0)),
                int(request.form.get("SupportCalls", 0)),
                int(request.form.get("PaymentDelay", 0)),
                1 if request.form.get("SubscriptionType", "Standard") == "Premium" else 0,
                int(request.form.get("ContractLength", 0)),
                float(request.form.get("TotalSpend", 0)),
                int(request.form.get("LastInteraction", 0)),
            ]
            churn_features = np.array(churn_features).reshape(1, -1)
            churn_features_scaled = scalers['churn_scaler'].transform(churn_features)

            predictions['churn_linear'] = "Churn" if models['churn_linear'].predict(churn_features_scaled)[0] else "No Churn"
            predictions['churn_rf'] = "Churn" if models['churn_rf'].predict(churn_features_scaled)[0] else "No Churn"

            # Spam Detection
            email_text = request.form.get("email_text", "")
            if email_text:
                email_tfidf = vectorizer.transform([email_text])
                predictions['spam_logistic'] = "Spam" if models['spam_logistic'].predict(email_tfidf)[0] else "Not Spam"
                predictions['spam_nb'] = "Spam" if models['spam_nb'].predict(email_tfidf)[0] else "Not Spam"

        except Exception as e:
            error = f"An error occurred: {str(e)}"

    return render_template("index.html", predictions=predictions, show_images=show_images, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)