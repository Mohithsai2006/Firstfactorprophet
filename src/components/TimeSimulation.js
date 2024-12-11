import React, { useState } from "react";
import "./TimeSimulation.css";

const TimeSimulation = () => {
  const [year, setYear] = useState(2028); // Default year
  const [forecastData, setForecastData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchForecast = async () => {
    setLoading(true);
    setError(null);

    // Check if the year is valid (>= 2021)
    if (year < 2021) {
      setError("Please select a year from 2021 onwards.");
      setLoading(false);
      return;
    }

    try {
      console.log("Sending request with year:", year); // Debugging log
      const response = await fetch("http://localhost:4000/forecast", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ year }), // This sends the selected year
      });

      if (!response.ok) {
        throw new Error("Failed to fetch forecast data");
      }

      const data = await response.json();
      console.log("Received response:", data); // Debugging log
      setForecastData(data);
    } catch (err) {
      console.error("Error fetching data:", err.message); // Debugging log
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="time-simulation">
      <h1>Time Simulation</h1>
      <div className="slider-container">
        <input
          type="range"
          min="2021"  // Ensure slider starts at 2021
          max="2050"
          value={year}
          onChange={(e) => setYear(Number(e.target.value))}  // Ensure it's a number
        />
        <p>Selected Year: {year}</p>
      </div>
      <button onClick={fetchForecast}>Get Forecast</button>

      {loading && <p>Loading...</p>}
      {error && <p>Error: {error}</p>}  {/* Display error if year is invalid */}
      {forecastData && (
        <div className="forecast-results">
          <h2>Forecast for {forecastData.year}</h2>
          <h3>Monthly Reservoir Levels</h3>
          <ul>
            {Object.entries(forecastData.monthly_level).map(([month, level]) => (
              <li key={month}>
                {month}: {level.toFixed(2)}
              </li>
            ))}
          </ul>
          <h3>Monthly Reservoir Storage</h3>
          <ul>
            {Object.entries(forecastData.monthly_storage).map(([month, storage]) => (
              <li key={month}>
                {month}: {storage.toFixed(2)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default TimeSimulation;
