# app/ml/predictor.py
import logging
from .yolo_analyzer import PlantAnalyzer
import os

logger = logging.getLogger(__name__)


class PlantPredictor:
    """Класс для работы с предсказаниями через API"""

    def __init__(self, model_path=None, pixels_per_mm=10.0):
        """
        Инициализация предиктора
        Args:
            model_path: путь к модели best.pt
            pixels_per_mm: коэффициент калибровки
        """
        if model_path is None:
            # Путь по умолчанию
            model_path = os.path.join(
                os.path.dirname(__file__),
                "models",
                "best.pt"
            )

        logger.info(f"Загрузка модели из {model_path}")
        self.analyzer = PlantAnalyzer(model_path, pixels_per_mm)
        logger.info("✅ Модель успешно загружена")

    def predict_from_bytes(self, image_bytes):
        """
        Предсказание из байтов изображения
        """
        try:
            result = self.analyzer.analyze_image(image_bytes)
            return result
        except Exception as e:
            logger.error(f"Ошибка при анализе: {e}")
            raise


# Создаем глобальный экземпляр предиктора
_predictor_instance = None


def get_predictor():
    """Получить экземпляр предиктора (синглтон)"""
    global _predictor_instance
    if _predictor_instance is None:
        # Путь к вашей модели
        model_path = os.path.join(
            os.path.dirname(__file__),
            "models",
            "best.pt"
        )
        _predictor_instance = PlantPredictor(
            model_path=model_path,
            pixels_per_mm=11.5  # Можно загружать из калибровки
        )
    return _predictor_instance