from flask import Flask, request, jsonify, render_template
from models.car_model import CarModel  # Import the CarModel logic
import threading
import os

app = Flask(__name__)

# Initialize the car model
car_model = CarModel()

@app.route('/')
def index():
    """
    Render the main HTML interface.
    """
    return render_template('index.html')

@app.route('/car_model', methods=['GET'])
def car_model_info():
    """
    Return details about the car model, such as training status and sample input format.
    """
    try:
        car_model_status = {
            "trained": os.path.exists(car_model.model_path),
            "expected_features": ["cylinders", "displacement", "weight", "acceleration", "model year"],
            "sample_input": "8,304,3433,12,70"
        }
        return jsonify(car_model_status)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/start-training-car', methods=['POST'])
def start_training_car():
    """
    Start training the car MPG model.
    """
    def train_thread():
        try:
            car_model.train()
            print("Car model training completed!")
        except Exception as e:
            print(f"Error during car model training: {e}")

    threading.Thread(target=train_thread).start()
    return jsonify({"message": "Car model training initiated!"})

@app.route('/predict-car-mpg', methods=['POST'])
def predict_car_mpg():
    """
    Predict MPG based on user input for the car model.
    """
    try:
        input_data = request.json.get('input', {})
        if not input_data:
            return jsonify({"error": "No input data provided."})
        car_name = car_model.predict(input_data)
        return jsonify({"car_name": car_name})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
