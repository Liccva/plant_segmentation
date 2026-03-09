# app/ml/yolo_analyzer.py
import torch
import cv2
import numpy as np
import json
import base64
from datetime import datetime
from ultralytics import YOLO
import os
from io import BytesIO
from PIL import Image
import logging
import time

logger = logging.getLogger(__name__)


class PlantAnalyzer:
    def __init__(self, model_path, pixels_per_mm=9.5):  # Изменено: 6.7 по умолчанию
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n{'=' * 50}")
        print(f"🚀 Инициализация PlantAnalyzer")
        print(f"{'=' * 50}")
        print(f"📊 Устройство: {self.device.upper()}")
        print(f"📏 Калибровка: {pixels_per_mm} пикс/мм")
        print(f"📁 Модель: {model_path}")

        start_time = time.time()
        self.model = YOLO(model_path)
        if self.device == 'cuda':
            self.model.to('cuda')
        else:
            self.model.model.eval()
        load_time = time.time() - start_time
        print(f"✅ Модель загружена за {load_time:.2f}с")

        self.original_pixels_per_mm = pixels_per_mm  # Сохраняем оригинальную калибровку
        self.pixels_per_mm = pixels_per_mm  # Будет обновляться с учетом масштаба

        # Истинные имена классов в модели (как они называются в файле модели)
        self.model_class_names = {
            0: 'root',  # в модели это root, но на самом деле лист
            1: 'stem',  # в модели это stem, но на самом деле корень
            2: 'leaf'  # в модели это leaf, но на самом деле стебель
        }

        # Соответствие для фронтенда (биологическая реальность)
        self.frontend_class_mapping = {
            0: 'leaf',  # class 0 (root) → лист
            1: 'root',  # class 1 (stem) → корень
            2: 'stem'  # class 2 (leaf) → стебель
        }

        # Цвета для визуализации (ключи – имена для фронтенда)
        self.colors = {
            'leaf': (255, 215, 0, 180),  # жёлтый для листьев
            'root': (155, 89, 182, 180),  # фиолетовый для корней
            'stem': (46, 204, 113, 180)  # зелёный для стеблей
        }

        print(f"📋 Классы модели (внутренние): {self.model_class_names}")
        print(f"📋 Соответствие для фронтенда: {self.frontend_class_mapping}")
        print(f"{'=' * 50}\n")

    # ---------- методы вычислений ----------
    def calculate_area_mm2(self, mask):
        """Площадь в мм²"""
        if isinstance(mask, list):
            mask = np.array(mask)
        area_pixels = np.count_nonzero(mask)
        area_mm2 = area_pixels / (self.pixels_per_mm ** 2)
        return round(area_mm2, 2)

    def calculate_length_mm(self, mask):
        """Длина через скелетонизацию (для корней и стеблей)"""
        if isinstance(mask, list):
            mask = np.array(mask)
        if np.count_nonzero(mask) == 0:
            return 0.0
        skeleton = self.skeletonize(mask)
        length_pixels = np.count_nonzero(skeleton)
        length_mm = length_pixels / self.pixels_per_mm
        return round(length_mm, 2)

    def skeletonize(self, mask):
        """Упрощенная скелетонизация через морфологию"""
        if isinstance(mask, list):
            mask = np.array(mask, dtype=np.uint8)
        else:
            mask = mask.astype(np.uint8)

        skeleton = np.zeros_like(mask)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            eroded = cv2.erode(mask, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(mask, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            mask = eroded.copy()
            if np.count_nonzero(mask) == 0:
                break
        return skeleton

    def mask_to_polygon(self, mask):
        """Конвертирует маску в полигон"""
        if isinstance(mask, list):
            mask = np.array(mask)
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.001 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            return approx.reshape(-1, 2).tolist()
        return []

    def create_mask_image(self, mask, image_size):
        """Создает изображение маски из массива маски YOLO (base64)"""
        try:
            if isinstance(mask, list):
                mask = np.array(mask)
            mask_normalized = (mask * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask_normalized, image_size, interpolation=cv2.INTER_NEAREST)
            mask_pil = Image.fromarray(mask_resized)
            buffered = BytesIO()
            mask_pil.save(buffered, format="PNG")
            mask_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return mask_base64
        except Exception as e:
            logger.error(f"Ошибка создания маски: {e}")
            return ""

    def create_overlay_image(self, image, masks_data):
        """Создает изображение с наложенными масками (base64)"""
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb).convert('RGBA')
            overlay = pil_image.copy()

            for obj in masks_data:
                if 'mask_binary' in obj:
                    mask_binary = np.array(obj['mask_binary']) if isinstance(obj['mask_binary'], list) else obj[
                        'mask_binary']
                    frontend_class = obj['frontend_class']
                    color = self.colors.get(frontend_class, (255, 0, 0, 180))
                    mask_rgba = np.zeros((mask_binary.shape[0], mask_binary.shape[1], 4), dtype=np.uint8)
                    mask_rgba[mask_binary > 0] = color
                    mask_pil = Image.fromarray(mask_rgba)
                    overlay = Image.alpha_composite(overlay, mask_pil)

            buffered = BytesIO()
            overlay.save(buffered, format="PNG")
            overlay_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return overlay_base64
        except Exception as e:
            logger.error(f"Ошибка создания overlay: {e}")
            return ""

    # ---------- основной метод анализа ----------
    def analyze_image(self, image_bytes):
        total_start = time.time()
        print(f"\n{'=' * 50}")
        print(f"🔍 АНАЛИЗ ИЗОБРАЖЕНИЯ")
        print(f"{'=' * 50}")

        t0 = time.time()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        original_height, original_width = img.shape[:2]
        print(f"📸 Оригинал: {original_width}x{original_height}")

        # Масштабирование
        max_size = 960
        scale = 1.0
        if max(original_width, original_height) > max_size:
            scale = max_size / max(original_width, original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"📏 Масштабирование: {new_width}x{new_height}, scale={scale:.3f}")

            # КОРРЕКЦИЯ: обновляем pixels_per_mm с учетом масштаба
            self.pixels_per_mm = self.original_pixels_per_mm * scale
            print(f"📏 Скорректированная калибровка: {self.pixels_per_mm:.2f} пикс/мм")
        else:
            self.pixels_per_mm = self.original_pixels_per_mm

        img_height, img_width = img.shape[:2]
        t1 = time.time()
        print(f"⏱️ Предобработка: {(t1 - t0) * 1000:.1f}мс")

        t2 = time.time()
        use_half = (self.device == 'cuda')
        results = self.model(img, conf=0.25, iou=0.45, half=use_half, verbose=False)
        t3 = time.time()
        inference_time = (t3 - t2) * 1000
        print(f"⏱️ Инференс: {inference_time:.1f}мс")

        result = results[0]

        # Подсчёт объектов по классам модели
        model_class_counts = {0: 0, 1: 0, 2: 0}
        if result.boxes is not None:
            classes = result.boxes.cls.cpu().numpy()
            print(f"🔍 Сырые class_id из модели: {classes}")
            for cls_id in classes:
                cls_id = int(cls_id)
                if cls_id in model_class_counts:
                    model_class_counts[cls_id] += 1

        print(f"\n📊 Обнаруженные объекты (классы модели):")
        print(f"   class 0 (root) = {model_class_counts[0]} объектов → это ЛИСТЬЯ")
        print(f"   class 1 (stem) = {model_class_counts[1]} объектов → это КОРНИ")
        print(f"   class 2 (leaf) = {model_class_counts[2]} объектов → это СТЕБЛИ")

        t4 = time.time()
        masks_data = []

        # Метрики по классам модели
        metrics = {
            'root': {'area_mm2': 0, 'length_mm': 0},
            'stem': {'area_mm2': 0, 'length_mm': 0},
            'leaf': {'area_mm2': 0, 'length_mm': 0}
        }

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy() if result.boxes is not None else []
            confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else []

            for i, mask in enumerate(masks):
                mask_resized = cv2.resize(mask.astype(np.uint8), (img_width, img_height))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)

                model_cls_id = int(class_ids[i]) if i < len(class_ids) else 0
                model_cls_name = self.model_class_names.get(model_cls_id, 'unknown')
                frontend_cls = self.frontend_class_mapping.get(model_cls_id, 'unknown')

                area = self.calculate_area_mm2(mask_binary)
                length = self.calculate_length_mm(mask_binary) if frontend_cls in ['root', 'stem'] else 0
                polygon = self.mask_to_polygon(mask)
                mask_base64 = self.create_mask_image(mask, (img_width, img_height))

                obj_data = {
                    'model_class': model_cls_name,
                    'model_class_id': model_cls_id,
                    'frontend_class': frontend_cls,
                    'area_mm2': area,
                    'length_mm': length,
                    'confidence': float(confidences[i]) if i < len(confidences) else 0.0,
                    'polygon': polygon,
                    'mask_base64': mask_base64,
                    'mask_binary': mask_binary.tolist()
                }
                masks_data.append(obj_data)

                if model_cls_name in metrics:
                    metrics[model_cls_name]['area_mm2'] += area
                    if frontend_cls in ['root', 'stem']:
                        metrics[model_cls_name]['length_mm'] += length

        t5 = time.time()
        overlay_base64 = ""
        if masks_data:
            overlay_base64 = self.create_overlay_image(img, masks_data)
        t6 = time.time()
        print(f"⏱️ Постобработка: {(t6 - t4) * 1000:.1f}мс")

        # Формируем результат для фронтенда с правильными ключами
        result_data = {
            'leaf': {
                'polygon': [],
                'mask': '',
                'length': metrics['root']['length_mm'],
                'area': metrics['root']['area_mm2'],
                'confidence': 0,
                'count': model_class_counts[0]
            },
            'root': {
                'polygon': [],
                'mask': '',
                'length': metrics['stem']['length_mm'],
                'area': metrics['stem']['area_mm2'],
                'confidence': 0,
                'count': model_class_counts[1]
            },
            'stem': {
                'polygon': [],
                'mask': '',
                'length': metrics['leaf']['length_mm'],
                'area': metrics['leaf']['area_mm2'],
                'confidence': 0,
                'count': model_class_counts[2]
            }
        }

        # Заполняем маски и confidence для каждого frontend-класса
        for obj in masks_data:
            if obj['frontend_class'] == 'leaf' and not result_data['leaf']['mask']:
                result_data['leaf']['mask'] = obj['mask_base64']
                result_data['leaf']['confidence'] = obj['confidence']
                result_data['leaf']['polygon'] = obj['polygon']
            elif obj['frontend_class'] == 'root' and not result_data['root']['mask']:
                result_data['root']['mask'] = obj['mask_base64']
                result_data['root']['confidence'] = obj['confidence']
                result_data['root']['polygon'] = obj['polygon']
            elif obj['frontend_class'] == 'stem' and not result_data['stem']['mask']:
                result_data['stem']['mask'] = obj['mask_base64']
                result_data['stem']['confidence'] = obj['confidence']
                result_data['stem']['polygon'] = obj['polygon']

        result_data['overlay_image'] = overlay_base64
        result_data['objects_count'] = len(masks_data)
        result_data['class_counts'] = {
            'leaf': model_class_counts[0],
            'root': model_class_counts[1],
            'stem': model_class_counts[2]
        }
        result_data['all_objects'] = [
            {
                'class': obj['frontend_class'],
                'class_id': obj['model_class_id'],
                'area_mm2': obj['area_mm2'],
                'length_mm': obj['length_mm'],
                'confidence': obj['confidence']
            }
            for obj in masks_data
        ]
        result_data['original_size'] = {'width': original_width, 'height': original_height}
        result_data['processed_size'] = {'width': img_width, 'height': img_height}
        result_data['device'] = self.device
        result_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Добавляем информацию о калибровке для отладки
        result_data['calibration_info'] = {
            'original_pixels_per_mm': self.original_pixels_per_mm,
            'adjusted_pixels_per_mm': self.pixels_per_mm,
            'scale_factor': scale
        }

        print(f"\n📊 Статистика для фронтенда:")
        print(f"   leaf (листья) из class 0: площадь={result_data['leaf']['area']:.2f} мм²")
        print(f"   root (корни) из class 1: площадь={result_data['root']['area']:.2f} мм²")
        print(f"   stem (стебли) из class 2: площадь={result_data['stem']['area']:.2f} мм²")
        print(f"\n📏 Информация о калибровке:")
        print(f"   Оригинальная: {self.original_pixels_per_mm} пикс/мм")
        print(f"   Скорректированная: {self.pixels_per_mm:.2f} пикс/мм")
        print(f"   Коэффициент масштаба: {scale:.3f}")

        total_time = time.time() - total_start
        print(f"\n⏱️ Общее время: {total_time:.2f}с")
        print(f"{'=' * 50}\n")

        return result_data