import os
from flask import Flask, render_template
import tensorflow as tf
from io import StringIO
import h5py

# Initialize Flask app
app = Flask(__name__)

# Path to the models folder
models_folder = './models'
model_summaries = {}

# Function to inspect `.h5` file content
def inspect_h5_file(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            return f"File contents: {keys}"
    except Exception as e:
        return f"Error inspecting file: {str(e)}"

# Function to load models and capture their summaries
def load_model_summaries():
    for file_name in os.listdir(models_folder):
        if file_name.endswith('.h5'):
            model_path = os.path.join(models_folder, file_name)
            try:
                # Load the model and generate the summary
                model = tf.keras.models.load_model(model_path)
                summary_stream = StringIO()
                model.summary(print_fn=lambda x: summary_stream.write(x + "\n"))
                summary = summary_stream.getvalue()
                summary_stream.close()
                model_summaries[file_name] = summary
            except Exception as e:
                # Inspect `.h5` file for debugging if model loading fails
                file_info = inspect_h5_file(model_path)
                model_summaries[file_name] = (
                    f"Error loading model: {str(e)}\n"
                    f"Additional Info: {file_info}"
                )

# Load all model summaries at startup
load_model_summaries()

# Home page route
@app.route('/')
def index():
    # Pass model summaries to the HTML template
    return render_template('index.html', model_summaries=model_summaries)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
