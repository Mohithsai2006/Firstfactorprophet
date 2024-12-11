const express = require("express");
const axios = require("axios");
const cors = require("cors");

const app = express();
const FLASK_API_URL = "http://127.0.0.1:5000";

app.use(cors());
app.use(express.json());

app.post("/forecast", async (req, res) => {
  try {
    const { year } = req.body;

    if (!year) {
      return res.status(400).json({ error: "Missing 'year' in request body" });
    }

    const response = await axios.post(`${FLASK_API_URL}/forecast`, { year });
    res.json(response.data);
  } catch (error) {
    console.error("Error communicating with Flask API:", error.message);
    res.status(500).json({ error: "Internal Server Error", details: error.message });
  }
});

const PORT = 4000;
app.listen(PORT, () => {
  console.log(`Express server running on http://localhost:${PORT}`);
});
