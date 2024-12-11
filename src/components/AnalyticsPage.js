
// import React from "react";
// import { useLocation } from "react-router-dom";
// import "./AnalyticsPage.css";
// import {
//   LineChart,
//   Line,
//   XAxis,
//   YAxis,
//   CartesianGrid,
//   Tooltip,
//   ResponsiveContainer,
// } from "recharts";

// const data = [
//   { name: "Jan", level: 75 },
//   { name: "Feb", level: 78 },
//   { name: "Mar", level: 80 },
//   { name: "Apr", level: 76 },
//   { name: "May", level: 72 },
//   { name: "Jun", level: 70 },
//   { name: "Jul", level: 74 },
// ];

// const AnalyticsPage = () => {
//   const location = useLocation();
//   const { year, district } = location.state || {}; // Get year and district from state

//   return (
//     <div className="analytics-container">
//       <h1 className="analytics-title">Water Storage Analytics</h1>
//       <p className="analytics-year">
//         Year: <strong>{year || "N/A"}</strong> | District: <strong>{district || "N/A"}</strong>
//       </p>
//       <div className="analytics-metrics">
//         <div className="metric">
//           <p>Current Water Level</p>
//           <h2>79.0%</h2>
//           <p>Percentage of maximum capacity</p>
//         </div>
//         <div className="metric">
//           <p>Daily Water Usage</p>
//           <h2>2.70 ML</h2>
//           <p>Megalitres per day</p>
//         </div>
//         <div className="metric">
//           <p>Water Quality Index</p>
//           <h2>93/100</h2>
//           <p>Based on multiple parameters</p>
//         </div>
//       </div>
//       <div className="detailed-analysis">
//         <h2>Detailed Water Storage Analysis</h2>
//         <ResponsiveContainer width="90%" height={300}>
//           <LineChart data={data}>
//             <CartesianGrid strokeDasharray="3 3" />
//             <XAxis dataKey="name" />
//             <YAxis />
//             <Tooltip />
//             <Line
//               type="monotone"
//               dataKey="level"
//               stroke="#0056b3"
//               strokeWidth={2}
//             />
//           </LineChart>
//         </ResponsiveContainer>
//       </div>
//     </div>
//   );
// };

// export default AnalyticsPage;






import React, { useState } from "react";
import { useLocation } from "react-router-dom";
import "./AnalyticsPage.css";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const AnalyticsPage = () => {
  const location = useLocation();
  const { year, district } = location.state || {};

  // State for selected graph type
  const [graphType, setGraphType] = useState("weekly");

  // Data for weekly, monthly, and yearly
  const weeklyData = [
    { name: "Week 1", level: 75 },
    { name: "Week 2", level: 78 },
    { name: "Week 3", level: 80 },
    { name: "Week 4", level: 76 },
  ];

  const monthlyData = [
    { name: "Jan", level: 70 },
    { name: "Feb", level: 72 },
    { name: "Mar", level: 74 },
    { name: "Apr", level: 75 },
    { name: "May", level: 78 },
    { name: "Jun", level: 80 },
    { name: "Jul", level: 82 },
    { name: "Aug", level: 79 },
    { name: "Sep", level: 76 },
    { name: "Oct", level: 74 },
    { name: "Nov", level: 72 },
    { name: "Dec", level: 70 },
  ];

  const yearlyData = [
    { name: "2018", level: 60 },
    { name: "2019", level: 65 },
    { name: "2020", level: 70 },
    { name: "2021", level: 75 },
    { name: "2022", level: 78 },
    { name: "2023", level: 80 },
    { name: "2024", level: 82 },
    { name: "2025", level: 85 },
  ];

  // Select data based on graphType
  const selectedData =
    graphType === "weekly" ? weeklyData : graphType === "monthly" ? monthlyData : yearlyData;

  return (
    <div className="analytics-container">
      <h1 className="analytics-title">Water Storage Analytics</h1>
      <p className="analytics-year">
        Year: <strong>{year || "N/A"}</strong> | District: <strong>{district || "N/A"}</strong>
      </p>

      {/* Graph Type Toggle */}
      <div className="graph-toggle">
        <button
          className={`toggle-button ${graphType === "weekly" ? "active" : ""}`}
          onClick={() => setGraphType("weekly")}
        >
          Weekly
        </button>
        <button
          className={`toggle-button ${graphType === "monthly" ? "active" : ""}`}
          onClick={() => setGraphType("monthly")}
        >
          Monthly
        </button>
        <button
          className={`toggle-button ${graphType === "yearly" ? "active" : ""}`}
          onClick={() => setGraphType("yearly")}
        >
          Yearly
        </button>
      </div>

      {/* Graph Display */}
      <div className="detailed-analysis">
        <ResponsiveContainer width="90%" height={300}>
          <LineChart data={selectedData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="level" stroke="#0056b3" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default AnalyticsPage;
