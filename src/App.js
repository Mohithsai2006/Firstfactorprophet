import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Navbar from "./components/Navbar";
import About from "./components/About";
import Contact from "./components/Contact";
import Map from "./components/Map";
import Home from "./components/Home";
import AnalyticsPage from "./components/AnalyticsPage";
import TimeSimulation from "./components/TimeSimulation";

const App = () => {
  return ( 
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route path="/contact" element={<Contact />} />
        <Route path="/map" element={<Map />} />
        <Route path="/TimeSimulation" element={<TimeSimulation />} />
        <Route path="/Analytics" element={<AnalyticsPage />} />
      </Routes>
    </Router>
  );
};

export default App;
