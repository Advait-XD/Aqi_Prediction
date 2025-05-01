from flask import Blueprint, render_template, request
import pandas as pd
import joblib
import os

main = Blueprint("main", __name__)

# Load dataset and model
DATA_PATH = os.path.join(os.getcwd(), "city_day.csv")
df = pd.read_csv(DATA_PATH)
cities = sorted(df['City'].dropna().unique())

model = joblib.load("aqi_model.pkl")
scaler = joblib.load("aqi_scaler.pkl")


def get_bucket(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


@main.route("/", methods=["GET", "POST"])
def index():
    selected_city = None
    aqi_data = []
    avg_pollutants = {}
    predicted_aqi = None
    aqi_bucket = None

    if request.method == "POST":
        selected_city = request.form.get("city")
        predict = request.form.get("predict") == "true"

        city_df = df[df["City"] == selected_city].copy()
        city_df["Date"] = pd.to_datetime(city_df["Date"], errors="coerce")
        city_df = city_df.dropna(subset=["Date", "AQI"])
        city_df = city_df.sort_values("Date")

        city_df['Date'] = city_df['Date'].dt.strftime('%Y-%m-%d')
        aqi_data = city_df[["Date", "AQI"]].to_dict(orient="records")

        pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx',
                          'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
        avg_pollutants = city_df[pollutant_cols].mean().round(2).to_dict()

        if predict and not city_df[pollutant_cols].dropna().empty:
            latest = city_df[pollutant_cols].dropna(
            ).iloc[-1].values.reshape(1, -1)
            scaled = scaler.transform(latest)
            predicted_aqi = round(model.predict(scaled)[0], 2)
            aqi_bucket = get_bucket(predicted_aqi)

    return render_template("index.html", cities=cities, selected_city=selected_city,
                           aqi_data=aqi_data, avg_pollutants=avg_pollutants,
                           predicted_aqi=predicted_aqi, aqi_bucket=aqi_bucket)
