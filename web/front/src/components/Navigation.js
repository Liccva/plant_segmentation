// src/components/Navigation.js
import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { logout, isAuthenticated, getUser } from '../services/auth';
import './Navigation.css';

const Navigation = () => {
  const navigate = useNavigate();
  const authenticated = isAuthenticated();
  const user = getUser();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <nav className="navbar">
      <div className="nav-container">
        <Link to="/" className="nav-logo">
          Plant Segmentation
        </Link>
        <div className="nav-menu">
          {authenticated ? (
            <>
              <span className="nav-user">Привет, {user?.login}</span>
              <Link to="/upload" className="nav-link">Загрузка</Link>
              <button onClick={handleLogout} className="nav-link logout-btn">
                Выход
              </button>
            </>
          ) : (
            <>
              <Link to="/login" className="nav-link">Вход</Link>
              <Link to="/register" className="nav-link">Регистрация</Link>
            </>
          )}
        </div>
      </div>
    </nav>
  );
};

export default Navigation;