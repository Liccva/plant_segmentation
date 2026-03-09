// src/components/ImageUpload.js
import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
// getUser больше не нужен, так как user_id определяется по токену на бэкенде
import ResultDisplay from './ResultDisplay';
import './ImageUpload.css';

const ImageUpload = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    setFile(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError('');
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.bmp']
    },
    maxSize: 10485760, // 10MB
    maxFiles: 1
  });

  const handleUpload = async () => {
    if (!file) return;

    setProcessing(true);
    setError('');

    const formData = new FormData();
    formData.append('file', file);
    // user_id больше не добавляем – он будет взят из токена на сервере

    try {
      console.log("📤 Отправка файла:", file.name);

      const response = await axios.post('http://localhost:8000/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Authorization': `Bearer ${localStorage.getItem('token')}` // токен обязателен
        }
      });

      console.log("✅ Ответ от сервера:", response.data);
      setResult(response.data);

    } catch (err) {
      console.error("❌ Ошибка:", err);
      // Если ошибка 401 – возможно, пользователь не авторизован
      if (err.response?.status === 401) {
        setError('Необходимо авторизоваться. Пожалуйста, войдите в систему.');
      } else {
        setError(err.response?.data?.detail || 'Ошибка при обработке изображения');
      }
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="upload-container">
      <div className="upload-header">
        <h1>Анализ изображения растения</h1>
        <p>Загрузите фотографию для определения стеблей, листьев и корней</p>
      </div>

      <div className="upload-content">
        <div className="upload-area">
          {!preview ? (
            <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
              <input {...getInputProps()} />
              <div className="dropzone-content">
                <span className="upload-icon">📸</span>
                <p>Перетащите изображение сюда или кликните для выбора</p>
                <small>Поддерживаются форматы: JPG, PNG, BMP (до 10MB)</small>
              </div>
            </div>
          ) : (
            <div className="preview-section">
              <div className="preview-image-container">
                <img src={preview} alt="Preview" className="preview-image" />
                <button
                  className="change-image-btn"
                  onClick={() => {
                    setFile(null);
                    setPreview(null);
                    setResult(null);
                  }}
                >
                  Изменить изображение
                </button>
              </div>

              {!processing && !result && (
                <button
                  className="analyze-btn"
                  onClick={handleUpload}
                  disabled={processing}
                >
                  Анализировать изображение
                </button>
              )}

              {processing && (
                <div className="processing">
                  <div className="spinner"></div>
                  <p>Анализируем изображение... это может занять до 30 секунд</p>
                </div>
              )}

              {error && <div className="error-message">{error}</div>}
            </div>
          )}
        </div>

        {result && (
          <ResultDisplay
            result={result}
            image={preview}
          />
        )}
      </div>
    </div>
  );
};

export default ImageUpload;