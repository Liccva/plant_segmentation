import axios from 'axios';

const API_URL = 'http://localhost:8000'; // URL вашего FastAPI бэкенда

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Добавление токена к запросам
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

export const login = (login, password) =>
  api.post('/token', new URLSearchParams({ username: login, password }));

export const register = (userData) =>
  api.post('/users/', userData);

export const getUsers = () =>
  api.get('/users/');

export const createPrediction = (predictionData) =>
  api.post('/predictions/', predictionData);

export const getUserPredictions = (userId) =>
  api.get(`/predictions/user/${userId}`);

export default api;