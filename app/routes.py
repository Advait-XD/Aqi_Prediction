# app/routes.py
from flask import Blueprint, render_template, request
import pandas as pd
import joblib
import os

main = Blueprint("main", __name__)

# Load your city data
DATA_PATH = os.path.join(os.getcwd(), "city_day.csv")
df = pd.read_csv(DATA_PATH)
cities = sorted(df['City'].unique())

# Load your trained model and scaler
model = joblib.load("aqi_model.pkl")
scaler = joblib.load("aqi_scaler.pkl")


@main.route("/", methods=["GET", "POST"])
def index():
    selected_city = None
    aqi_data = []

    if request.method == "POST":
        selected_city = request.form.get("city")
        city_df = df[df["City"] == selected_city].copy()

        # Convert Date to datetime
        city_df['Date'] = pd.to_datetime(city_df['Date'], errors='coerce')
        city_df = city_df.dropna(subset=["Date"])
        city_df = city_df.sort_values("Date")

        # Drop rows without AQI
        city_df = city_df.dropna(subset=["AQI"])

        # Format Date and extract AQI for chart
        city_df['Date'] = city_df['Date'].dt.strftime('%Y-%m-%d')
        aqi_data = city_df[['Date', 'AQI']].to_dict(orient='records')

    return render_template("index.html", cities=cities, selected_city=selected_city, aqi_data=aqi_data)
