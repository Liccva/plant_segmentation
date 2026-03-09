// src/components/ResultDisplay.js
import React, { useRef, useEffect, useState } from 'react';
import './ResultDisplay.css';

const ResultDisplay = ({ result, image }) => {
  const canvasRef = useRef(null);
  const [activeMasks, setActiveMasks] = useState({
    root: true,
    stem: true,
    leaf: true
  });

  // Цвета для масок (в цвет плашек, прямое соответствие)
  const maskColors = {
    root: { r: 155, g: 89, b: 182, a: 0.6 },   // фиолетовый для корней
    stem: { r: 46, g: 204, b: 113, a: 0.6 },   // зеленый для стеблей
    leaf: { r: 241, g: 196, b: 15, a: 0.6 }    // желтый для листьев
  };

  useEffect(() => {
    if (!result?.data || !image) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = new Image();

    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;

      // Рисуем оригинал
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      // Функция для рисования маски
      const drawMask = (maskBase64, color, isActive) => {
        if (!maskBase64 || !isActive) return;

        const maskImg = new Image();
        maskImg.onload = () => {
          const maskCanvas = document.createElement('canvas');
          maskCanvas.width = canvas.width;
          maskCanvas.height = canvas.height;
          const maskCtx = maskCanvas.getContext('2d');
          maskCtx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);

          const maskData = maskCtx.getImageData(0, 0, canvas.width, canvas.height);
          const mainData = ctx.getImageData(0, 0, canvas.width, canvas.height);

          for (let i = 0; i < maskData.data.length; i += 4) {
            if (maskData.data[i] > 0) {
              // Яркое наложение цвета
              mainData.data[i] = Math.min(255, mainData.data[i] * 0.4 + color.r * 0.6);
              mainData.data[i + 1] = Math.min(255, mainData.data[i + 1] * 0.4 + color.g * 0.6);
              mainData.data[i + 2] = Math.min(255, mainData.data[i + 2] * 0.4 + color.b * 0.6);
            }
          }
          ctx.putImageData(mainData, 0, 0);
        };
        maskImg.src = `data:image/png;base64,${maskBase64}`;
      };

      const data = result.data;
      if (data.root?.mask && activeMasks.root) drawMask(data.root.mask, maskColors.root, true);
      if (data.stem?.mask && activeMasks.stem) drawMask(data.stem.mask, maskColors.stem, true);
      if (data.leaf?.mask && activeMasks.leaf) drawMask(data.leaf.mask, maskColors.leaf, true);

      // Добавляем подписи сверху (прямое соответствие)
      setTimeout(() => {
        ctx.font = 'bold 16px "Segoe UI", Arial, sans-serif';
        ctx.shadowColor = 'rgba(0, 0, 0, 0.7)';
        ctx.shadowBlur = 4;
        ctx.shadowOffsetX = 2;
        ctx.shadowOffsetY = 2;

        let yOffset = 10;

        if (data.root?.area > 0 && activeMasks.root) {
          ctx.fillStyle = '#9b59b6';
          ctx.fillRect(10, yOffset, 110, 30);
          ctx.fillStyle = '#ffffff';
          ctx.fillText('🌱 Корень', 15, yOffset + 22);
          yOffset += 40;
        }

        if (data.stem?.area > 0 && activeMasks.stem) {
          ctx.fillStyle = '#2ecc71';
          ctx.fillRect(10, yOffset, 120, 30);
          ctx.fillStyle = '#000000';
          ctx.fillText('🌿 Стебель', 15, yOffset + 22);
          yOffset += 40;
        }

        if (data.leaf?.area > 0 && activeMasks.leaf) {
          ctx.fillStyle = '#f1c40f';
          ctx.fillRect(10, yOffset, 110, 30);
          ctx.fillStyle = '#000000';
          ctx.fillText('🍃 Лист', 15, yOffset + 22);
        }

        ctx.shadowColor = 'transparent';
      }, 100);
    };
    img.src = image;
  }, [result, image, activeMasks]);

  const formatNumber = (num) => num?.toFixed(1) || '0';
  const formatConfidence = (conf) => conf ? `${Math.round(conf * 100)}%` : '0%';

  const toggleMask = (part) => {
    setActiveMasks(prev => ({ ...prev, [part]: !prev[part] }));
  };

  const data = result?.data || {};

  return (
    <div className="result-container">
      <div className="image-section">
        <h3>Результат анализа</h3>
        <div className="canvas-wrapper">
          <canvas ref={canvasRef} className="result-canvas" style={{ width: '100%' }} />
        </div>
      </div>

      <div className="stats-section">
        {data.root?.area > 0 && (
          <div className={`stat-item root-badge ${!activeMasks.root ? 'inactive' : ''}`}>
            <div className="stat-header">
              <input
                type="checkbox"
                checked={activeMasks.root}
                onChange={() => toggleMask('root')}
                className="mask-checkbox"
              />
              <span className="stat-icon">🌱</span>
              <span className="stat-name">Корень</span>
            </div>
            <div className="stat-values">
              <span className="stat-value">📏 {formatNumber(data.root.length)} мм</span>
              <span className="stat-value">📐 {formatNumber(data.root.area)} мм²</span>
              <span className="stat-value">🎯 {formatConfidence(data.root.confidence)}</span>
            </div>
          </div>
        )}

        {data.stem?.area > 0 && (
          <div className={`stat-item stem-badge ${!activeMasks.stem ? 'inactive' : ''}`}>
            <div className="stat-header">
              <input
                type="checkbox"
                checked={activeMasks.stem}
                onChange={() => toggleMask('stem')}
                className="mask-checkbox"
              />
              <span className="stat-icon">🌿</span>
              <span className="stat-name">Стебель</span>
            </div>
            <div className="stat-values">
              <span className="stat-value">📏 {formatNumber(data.stem.length)} мм</span>
              <span className="stat-value">📐 {formatNumber(data.stem.area)} мм²</span>
              <span className="stat-value">🎯 {formatConfidence(data.stem.confidence)}</span>
            </div>
          </div>
        )}

        {data.leaf?.area > 0 && (
          <div className={`stat-item leaf-badge ${!activeMasks.leaf ? 'inactive' : ''}`}>
            <div className="stat-header">
              <input
                type="checkbox"
                checked={activeMasks.leaf}
                onChange={() => toggleMask('leaf')}
                className="mask-checkbox"
              />
              <span className="stat-icon">🍃</span>
              <span className="stat-name">Лист</span>
            </div>
            <div className="stat-values">
              <span className="stat-value">📏 {formatNumber(data.leaf.length)} мм</span>
              <span className="stat-value">📐 {formatNumber(data.leaf.area)} мм²</span>
              <span className="stat-value">🎯 {formatConfidence(data.leaf.confidence)}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ResultDisplay;