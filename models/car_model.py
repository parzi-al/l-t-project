import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import os

class CarModel:
    def __init__(self, dataset_path='auto-mpg.csv', model_path='car_model.keras'):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.model = None
        self.car_name_map = None

    def load_dataset(self):
        """
        Load the dataset and preprocess it for training.
        """
        df = pd.read_csv(self.dataset_path, na_values=['NA', '?'])
        self.car_name_map = df[['cylinders', 'displacement', 'weight', 'acceleration', 'model year', 'car name']]
        df.drop(columns=['car name'], inplace=True)
        df.dropna(inplace=True)
        return df

    def preprocess_data(self, df):
        """
        Preprocess the data: normalize features and split into training and testing sets.
        """
        X = df.drop(columns=['mpg', 'horsepower'])
        y = df['mpg']
        X = (X - X.mean()) / X.std()
        return train_test_split(X, y, test_size=0.25, random_state=42)

    def define_model(self, input_dim):
        """
        Define the neural network architecture.
        """
        model = Sequential([
            Dense(25, input_dim=input_dim, activation='relu'),
            Dense(10, activation='relu'),
            Dense(1)
        ])
        model.compile(loss="mean_squared_error", optimizer='adam')
        return model

    def train(self, epochs=100, batch_size=32):
        """
        Train the car MPG model and save it.
        """
        df = self.load_dataset()
        X_train, X_test, y_train, y_test = self.preprocess_data(df)
        self.model = self.define_model(input_dim=X_train.shape[1])
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        self.model.save(self.model_path)

    def predict(self, input_data):
        """
        Predict the car name based on input features.
        """
        if self.model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError("Model not found. Train the model first!")
            self.model = load_model(self.model_path)

        car_name_df = self.car_name_map.copy()
        input_df = pd.DataFrame([input_data])
        car_name_df['distance'] = ((car_name_df[['cylinders', 'displacement', 'weight', 'acceleration', 'model year']] - input_df) ** 2).sum(axis=1)
        closest_row = car_name_df.loc[car_name_df['distance'].idxmin()]
        return closest_row['car name']
