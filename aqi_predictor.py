import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

class AQIPredictor:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.model = RandomForestRegressor()
        self.scaler = StandardScaler()
        self.feature_names = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
        self.target_name = 'AQI'  # Assuming you have an AQI column in your dataset

    def prepare_data(self):
        # Prepare the data for training
        X = self.data[self.feature_names]
        y = self.data[self.target_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train = self.scaler.fit_transform(X_train)
        self.y_train = y_train
        self.X_test = self.scaler.transform(X_test)
        self.y_test = y_test

    def train(self):
        self.prepare_data()
        self.model.fit(self.X_train, self.y_train)

    def predict(self, new_data):
        new_data_scaled = self.scaler.transform(new_data)
        return self.model.predict(new_data_scaled)

    def save_model(self, model_path, scaler_path):
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    def load_model(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)