from flask import Flask, render_template, request
import joblib
import os
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(STATIC_DIR, exist_ok=True)

# Load Models
def load_model(file_name):
    model_path = os.path.join(MODEL_DIR, file_name)
    return joblib.load(model_path) if os.path.exists(model_path) else None

models = {name: load_model(f"{name}.pkl") for name in [
    "digits_pca_model", "digits_tsne_model", "house_linear_regression_model",
    "house_random_forest_model", "cust_Logistic_Regression", "cust_Random_Forest",
    "spam_logistic_regression", "spam_naive_bayes"
]}

scalers = {"churn_scaler": load_model("cust_scaler.pkl")}
vectorizer = load_model("spam_vectorizer.pkl")
digits = load_digits()

# Plot Helper Functions
def save_plot(filename, plot_func, *args, **kwargs):
    filepath = os.path.join(STATIC_DIR, filename)
    if not os.path.exists(filepath):
        plot_func(*args, **kwargs)
        plt.savefig(filepath)
        plt.close()

@app.route("/", methods=["GET", "POST"])
def index():
    predictions, show_images, error = {}, False, None

    try:
        if request.method == "POST":
            form_data = request.form

            if "generate_plots" in form_data:
                pca_data = models['digits_pca_model'].transform(digits.data)
                save_plot("pca_plot.png", sns.scatterplot, x=pca_data[:, 0], y=pca_data[:, 1], hue=digits.target, palette="viridis")
                tsne_data = models['digits_tsne_model'].fit_transform(digits.data)
                save_plot("tsne_plot.png", sns.scatterplot, x=tsne_data[:, 0], y=tsne_data[:, 1], hue=digits.target, palette="coolwarm")
                show_images = True

            # House Prediction
            if "LotArea" in form_data:
                house_features = np.array([[
                    float(form_data.get("LotArea", 0)),
                    int(form_data.get("OverallQual", 0)),
                    int(form_data.get("OverallCond", 0)),
                    float(form_data.get("GrLivArea", 0)),
                    int(form_data.get("BedroomAbvGr", 0)),
                    int(form_data.get("YearBuilt", 0)),
                ]])
                predictions['house_linear'] = models['house_linear_regression_model'].predict(house_features)[0]
                predictions['house_rf'] = models['house_random_forest_model'].predict(house_features)[0]

            # Churn Prediction
            if "Age" in form_data:
                churn_features = np.array([[
                    float(form_data.get("Age", 0)),
                    1 if form_data.get("Gender", "Male") == "Female" else 0,
                    int(form_data.get("Tenure", 0)),
                    int(form_data.get("UsageFrequency", 0)),
                    int(form_data.get("SupportCalls", 0)),
                    int(form_data.get("PaymentDelay", 0)),
                    1 if form_data.get("SubscriptionType", "Standard") == "Premium" else 0,
                    int(form_data.get("ContractLength", 0)),
                    float(form_data.get("TotalSpend", 0)),
                    int(form_data.get("LastInteraction", 0)),
                ]])
                scaled_features = scalers['churn_scaler'].transform(churn_features)
                predictions['churn_linear'] = "Churn" if models['cust_Logistic_Regression'].predict(scaled_features)[0] else "No Churn"
                predictions['churn_rf'] = "Churn" if models['cust_Random_Forest'].predict(scaled_features)[0] else "No Churn"

            # Spam Detection
            if "email_text" in form_data:
                email_tfidf = vectorizer.transform([form_data["email_text"]])
                predictions['spam_logistic'] = "Spam" if models['spam_logistic_regression'].predict(email_tfidf)[0] else "Not Spam"
                predictions['spam_nb'] = "Spam" if models['spam_naive_bayes'].predict(email_tfidf)[0] else "Not Spam"

    except Exception as e:
        error = str(e)

    return render_template("index.html", predictions=predictions, show_images=show_images, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
