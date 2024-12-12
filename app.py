from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load the dataset
DATA_PATH = "GERUSOPPAORIGINAL.xlsx"  # Adjust path to your data file
df = pd.read_excel(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])

# Normalize the data for LSTM
scaler_level = MinMaxScaler(feature_range=(0, 1))
scaler_storage = MinMaxScaler(feature_range=(0, 1))

level_data_scaled = scaler_level.fit_transform(df[['Level']])
storage_data_scaled = scaler_storage.fit_transform(df[['Storage']])

# Function to create sequences for LSTM input
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Parameters for the LSTM model
time_steps = 30  # Use past 30 days to predict the next day

# Create sequences for level and storage data
X_level, y_level = create_sequences(level_data_scaled, time_steps)
X_storage, y_storage = create_sequences(storage_data_scaled, time_steps)

# Split the data into training and testing sets
X_level_train, X_level_test, y_level_train, y_level_test = train_test_split(X_level, y_level, test_size=0.2, shuffle=False)
X_storage_train, X_storage_test, y_storage_train, y_storage_test = train_test_split(X_storage, y_storage, test_size=0.2, shuffle=False)

# Define the BiLSTM model
def create_bilstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=False, input_shape=input_shape)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Create and train the BiLSTM models for Level and Storage
model_level = create_bilstm_model((time_steps, 1))
model_level.fit(X_level_train, y_level_train, epochs=10, batch_size=32, verbose=1)

model_storage = create_bilstm_model((time_steps, 1))
model_storage.fit(X_storage_train, y_storage_train, epochs=10, batch_size=32, verbose=1)
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

        # Prepare arrays for storing predictions
        future_level_scaled = []
        future_storage_scaled = []

        # Start with the last 30 days of data to predict the future
        current_level_sequence = level_data_scaled[-time_steps:].reshape(1, time_steps, 1)
        current_storage_sequence = storage_data_scaled[-time_steps:].reshape(1, time_steps, 1)

        # Predict for each future day
        for _ in future_dates:
            predicted_level = model_level.predict(current_level_sequence)
            predicted_storage = model_storage.predict(current_storage_sequence)

            # Append the prediction to the result
            future_level_scaled.append(predicted_level[0][0])
            future_storage_scaled.append(predicted_storage[0][0])

            # Update the sequence to include the predicted values
            current_level_sequence = np.append(current_level_sequence[:, 1:, :], predicted_level.reshape(1, 1, 1), axis=1)
            current_storage_sequence = np.append(current_storage_sequence[:, 1:, :], predicted_storage.reshape(1, 1, 1), axis=1)

        # Inverse transform to get the original scale
        future_level = scaler_level.inverse_transform(np.array(future_level_scaled).reshape(-1, 1))
        future_storage = scaler_storage.inverse_transform(np.array(future_storage_scaled).reshape(-1, 1))

        # Create Series for future predictions
        future_level_series = pd.Series(future_level.flatten(), index=future_dates)
        future_storage_series = pd.Series(future_storage.flatten(), index=future_dates)

        # Monthly aggregation
        forecast_level_series_monthly = future_level_series.resample('M').mean()
        forecast_storage_series_monthly = future_storage_series.resample('M').mean()

        # Convert Year-Month Period to string for response
        monthly_level_str = {str(key): value for key, value in forecast_level_series_monthly.items()}
        
        # Ensure that storage values retain all decimal places using repr
        monthly_storage_str = {str(key): repr(value) for key, value in forecast_storage_series_monthly.items()}

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
