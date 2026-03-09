# app/ml/model.py
import numpy as np
from typing import Dict, List, Tuple
import json
from PIL import Image
import os
import base64
from io import BytesIO


class PlantAnalysisModel:
    """
    Класс для вашей кастомной модели анализа растений
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__),
            "models",
            "your_model.pth"
        )
        self.model = None
        self.device = None
        self.load_model()

    def load_model(self):
        """Загрузка вашей реальной модели"""
        print(f"Загрузка модели из {self.model_path}")

        # TODO: РАСКОММЕНТИРУЙТЕ НУЖНЫЙ ВАРИАНТ
        # Здесь загружается ваша модель

        # Если модель не загружена, используем заглушку
        if self.model is None:
            print("⚠️ Модель не загружена, используется заглушка!")

    def preprocess(self, image: Image.Image):
        """Предобработка изображения"""
        # TODO: ваша предобработка
        return image

    def predict(self, input_data) -> Dict:
        """
        Основной метод предсказания с масками
        """
        if self.model is None:
            return self._dummy_predict_with_masks()

        # TODO: ЗДЕСЬ ВЫЗОВ ВАШЕЙ РЕАЛЬНОЙ МОДЕЛИ
        # Модель должна вернуть полигоны и маски

        return self._dummy_predict_with_masks()

    def _create_mask_image(self, polygon, image_size=(224, 224)):
        """Создание изображения маски из полигона"""
        from PIL import Image, ImageDraw

        # Создаем новое изображение
        mask = Image.new('L', image_size, 0)
        draw = ImageDraw.Draw(mask)

        # Рисуем полигон
        if polygon and len(polygon) > 2:
            draw.polygon(polygon, fill=255)

        # Конвертируем в base64
        buffered = BytesIO()
        mask.save(buffered, format="PNG")
        mask_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return mask_base64

    def _create_overlay_image(self, original_image, polygons, colors):
        """Создание изображения с наложенными масками"""
        from PIL import Image, ImageDraw

        # Конвертируем оригинал в RGB если нужно
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')

        # Создаем копию для рисования
        overlay = original_image.copy()
        draw = ImageDraw.Draw(overlay, 'RGBA')

        # Рисуем каждый полигон с полупрозрачным цветом
        for part, polygon in polygons.items():
            if polygon and len(polygon) > 2:
                color = colors.get(part, (255, 0, 0, 128))
                draw.polygon(polygon, fill=color)

        # Конвертируем в base64
        buffered = BytesIO()
        overlay.save(buffered, format="PNG")
        overlay_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return overlay_base64

    def _dummy_predict_with_masks(self):
        """Заглушка с масками для тестирования"""
        import random
        from datetime import datetime

        # Генерируем случайные полигоны
        root_polygon = [[50, 150], [70, 180], [90, 170], [80, 140]]
        stem_polygon = [[100, 100], [120, 80], [140, 100], [120, 120]]
        leaf_polygon = [[150, 50], [170, 30], [190, 50], [170, 70]]

        # Создаем маски
        root_mask = self._create_mask_image(root_polygon)
        stem_mask = self._create_mask_image(stem_polygon)
        leaf_mask = self._create_mask_image(leaf_polygon)

        # Цвета для наложения (R,G,B, Alpha)
        colors = {
            'root': (139, 69, 19, 128),  # коричневый
            'stem': (34, 139, 34, 128),  # зеленый
            'leaf': (50, 205, 50, 128)  # светло-зеленый
        }

        # Создаем тестовое изображение для наложения
        test_image = Image.new('RGB', (224, 224), color='white')
        overlay = self._create_overlay_image(
            test_image,
            {
                'root': root_polygon,
                'stem': stem_polygon,
                'leaf': leaf_polygon
            },
            colors
        )

        return {
            'root': {
                'polygon': root_polygon,
                'mask': root_mask,
                'length': round(random.uniform(10, 20), 1),
                'area': round(random.uniform(30, 50), 1),
                'confidence': round(random.uniform(0.85, 0.98), 2)
            },
            'stem': {
                'polygon': stem_polygon,
                'mask': stem_mask,
                'length': round(random.uniform(20, 30), 1),
                'area': round(random.uniform(25, 40), 1),
                'confidence': round(random.uniform(0.85, 0.98), 2)
            },
            'leaf': {
                'polygon': leaf_polygon,
                'mask': leaf_mask,
                'length': round(random.uniform(15, 25), 1),
                'area': round(random.uniform(45, 65), 1),
                'confidence': round(random.uniform(0.85, 0.98), 2)
            },
            'overlay_image': overlay,
            'timestamp': datetime.utcnow().isoformat()
        }


# Создаем глобальный экземпляр модели
_model_instance = None


def get_model():
    """Получить экземпляр модели (синглтон)"""
    global _model_instance
    if _model_instance is None:
        _model_instance = PlantAnalysisModel()
    return _model_instance