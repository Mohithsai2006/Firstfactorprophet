from flask import Flask, request, jsonify
import pandas as pd
from prophet import Prophet
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load the dataset
DATA_PATH = "GERUSOPPAORIGINAL.xlsx"  # Adjust path to your data file
df = pd.read_excel(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])

# Prepare data for Prophet
level_data = df[['Date', 'Level']].rename(columns={'Date': 'ds', 'Level': 'y'})
storage_data = df[['Date', 'Storage']].rename(columns={'Date': 'ds', 'Storage': 'y'})

# Train models for Level and Storage
model_level = Prophet(yearly_seasonality=True, daily_seasonality=True)
model_level.fit(level_data)

model_storage = Prophet(yearly_seasonality=True, daily_seasonality=True)
model_storage.fit(storage_data)

@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        # Receive the year from the request body
        data = request.json
        if not data or "year" not in data:
            return jsonify({"error": "Missing 'year' in request data"}), 400

        year = int(data.get("year"))
        print(f"Forecast requested for year: {year}")  # Debugging log

        # Validate year range to allow only years greater than 2020
        if year < 2021:
            return jsonify({"error": "Forecasting is only available for years after 2020"}), 400

        # Validate year range
        if year < 2000 or year > 2050:
            return jsonify({"error": "Year must be between 2000 and 2050"}), 400

        # Generate future dates for the given year
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        future_dates = pd.date_range(start=start_date, end=end_date)

        # Predictions for level and storage
        future_level = pd.DataFrame({"ds": future_dates})
        future_storage = pd.DataFrame({"ds": future_dates})

        forecast_level = model_level.predict(future_level)
        forecast_storage = model_storage.predict(future_storage)

        # Monthly aggregation
        forecast_level["Year-Month"] = forecast_level["ds"].dt.to_period("M")
        forecast_storage["Year-Month"] = forecast_storage["ds"].dt.to_period("M")

        monthly_level = forecast_level.groupby("Year-Month")["yhat"].mean()
        monthly_storage = forecast_storage.groupby("Year-Month")["yhat"].mean()

        # Convert Year-Month Period to string for response
        monthly_level_str = {str(key): value for key, value in monthly_level.items()}
        monthly_storage_str = {str(key): value for key, value in monthly_storage.items()}

        # Prepare response data
        response = {
            "year": year,
            "monthly_level": monthly_level_str,
            "monthly_storage": monthly_storage_str,
        }
        return jsonify(response)

    except Exception as e:
        print("Error processing the request:", e)  # Debugging log
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
