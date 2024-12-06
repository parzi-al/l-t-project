from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import threading
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import os

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
model = None
car_name_map = None

# Function to preprocess input data for prediction
def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data])
    required_features = ["cylinders", "displacement", "weight", "acceleration", "model year"]
    input_df = input_df[required_features]
    input_df = (input_df - input_df.mean()) / input_df.std()
    return input_df

# Function to train the model
def train_model():
    global model, car_name_map
    try:
        df = pd.read_csv('auto-mpg.csv', na_values=['NA', '?'])
        car_name_map = df[['cylinders', 'displacement', 'weight', 'acceleration', 'model year', 'car name']]
        df.drop(columns=['car name'], inplace=True)
        df.dropna(inplace=True)
        X = df.drop(columns=['mpg', 'horsepower'])
        y = df['mpg']
        X = (X - X.mean()) / X.std()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        model = Sequential([
            Dense(25, input_dim=X_train.shape[1], activation='relu'),
            Dense(10, activation='relu'),
            Dense(1)
        ])
        model.compile(loss="mean_squared_error", optimizer='adam')

        epochs = 1000
        for epoch in range(epochs):
            history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
            socketio.emit('training_update', {
                'epoch': epoch + 1,
                'total_epochs': epochs,
                'loss': history.history['loss'][-1]
            })

        model.save('model.keras')
        socketio.emit('training_complete', {'message': 'Training completed successfully!'})
    except Exception as e:
        socketio.emit('training_error', {'error': str(e)})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start-training', methods=['POST'])
def start_training():
    threading.Thread(target=train_model).start()
    return jsonify({"message": "Training started!"})

@app.route('/predict', methods=['POST'])
def predict():
    global model, car_name_map
    try:
        if model is None:
            model_path = 'model.keras'
            if not os.path.exists(model_path):
                return jsonify({"error": "Model not found. Train the model first!"})
            model = load_model(model_path)

        input_data = request.json.get('input', {})
        if not input_data:
            return jsonify({"error": "No input data provided."})

        input_df = preprocess_input(input_data)
        prediction = model.predict(input_df)[0][0]

        car_name_df = car_name_map.copy()
        car_name_df['distance'] = ((car_name_df[['cylinders', 'displacement', 'weight', 'acceleration', 'model year']] - pd.DataFrame([input_data])) ** 2).sum(axis=1)
        closest_row = car_name_df.loc[car_name_df['distance'].idxmin()]
        car_name = closest_row['car name']

        return jsonify({"mpg_prediction": prediction, "car_name": car_name})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=True)
