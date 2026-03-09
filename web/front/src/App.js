// src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Navigation from './components/Navigation';
import Login from './components/Login';
import Register from './components/Register';
import ImageUpload from './components/ImageUpload';
import { isAuthenticated } from './services/auth';
import './App.css';

const PrivateRoute = ({ children }) => {
  return isAuthenticated() ? children : <Navigate to="/login" />;
};

function App() {
  return (
    <Router>
      <div className="App">
        <Navigation />
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route
            path="/upload"
            element={
              <PrivateRoute>
                <ImageUpload />
              </PrivateRoute>
            }
          />
          <Route path="/" element={<Navigate to="/upload" />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;