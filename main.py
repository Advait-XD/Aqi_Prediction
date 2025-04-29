import pandas as pd
import numpy as np
import os
import logging
from aqi_predictor import AQIPredictor
from aqi_calculator import calculate_aqi
from grey_wolf_optimizer import GreyWolfOptimizer

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize predictor
    # Update the path to your CSV file
    data_path = r"C:\Advait_projects\AQI_Predictor-master\city_day.csv"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        exit(1)

    predictor = AQIPredictor(data_path)

    # Train the model
    predictor.train()
    logger.info("Model trained successfully.")

    # Example of making predictions on new data
    new_data = pd.DataFrame({
        'PM2.5': [15.0],
        'PM10': [45.0],
        'NO': [5.0],
        'NO2': [40.0],
        'NOx': [50.0],
        'NH3': [10.0],
        'CO': [3.5],
        'SO2': [25.0],
        'O3': [0.045],
        'Benzene': [0.005],
        'Toluene': [0.003],
        'Xylene': [0.002]
    })

    try:
        prediction = predictor.predict(new_data)
        logger.info(f"Predicted AQI for new data: {prediction[0]:.2f}")
    except Exception as e:
        logger.error(f"Error making predictions: {e}")

    # Save the model
    try:
        predictor.save_model("aqi_model.pkl", "aqi_scaler.pkl")
        logger.info("Model and scalers saved successfully.")
    except Exception as e:
        logger.error(f"Error saving model: {e}")

    # Calculate overall AQI from the dataset
    overall_aqi = calculate_aqi(predictor.data)
    logger.info(f"Overall AQI from dataset: {overall_aqi}")

    # Example of using Grey Wolf Optimizer
    def objective_function(params):
        # Define your objective function here
        # For example, you could minimize the error of the AQI prediction
        return np.sum(params**2)  # Placeholder for actual objective function

    lb = [0] * len(predictor.feature_names)  # Lower bounds
    ub = [100] * len(predictor.feature_names)  # Upper bounds
    gwo = GreyWolfOptimizer(objective_function, lb, ub,
                            dim=len(predictor.feature_names))
    best_params, best_score, convergence_curve = gwo.optimize()
    logger.info(f"Best parameters found by GWO: {best_params}")
    logger.info(f"Best score: {best_score}")
