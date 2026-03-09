import os
import json
import glob
import tempfile
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from telegram import Update, ForceReply
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ===================== НАСТРОЙКИ =====================
BOT_TOKEN = os.environ.get("BOT_TOKEN", "ВАШ_ТОКЕН")  # замените на свой токен
MODEL_PATH = "runs/segment/runs/segment/baseline_s_1024_20260309_012026/weights/best.pt"         # путь к модели YOLO
DEFAULT_PIXELS_PER_MM = 10.0                                     # значение по умолчанию
CALIB_SQUARE_SIZE_MM = 10.0                                      # размер клетки шахматки в мм
TEMP_DIR = "temp"                                                 # папка для временных файлов

# Логирование
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# ===================== JSON ENCODER ДЛЯ NUMPY =====================
class NumpyEncoder(json.JSONEncoder):
    """Сериализация numpy типов в JSON"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ===================== КЛАСС АНАЛИЗАТОРА =====================
class PlantAnalyzer:
    def __init__(self, model_path, default_pixels_per_mm=10.0):
        """Загружает модель YOLO и задаёт коэффициент по умолчанию"""
        self.model = YOLO(model_path)
        self.default_pixels_per_mm = default_pixels_per_mm
        self.class_names = {0: 'root', 1: 'stem', 2: 'leaf'}

    def calculate_area_mm2(self, mask, pixels_per_mm):
        """Площадь в мм²"""
        area_pixels = np.count_nonzero(mask)
        area_mm2 = area_pixels / (pixels_per_mm ** 2)
        return float(round(area_mm2, 2))

    def calculate_length_mm(self, mask, pixels_per_mm):
        """Длина через скелетонизацию (для корней и стеблей)"""
        if np.count_nonzero(mask) == 0:
            return 0.0

        # Скелетонизация
        skeleton = self.skeletonize(mask)
        length_pixels = np.count_nonzero(skeleton)
        length_mm = length_pixels / pixels_per_mm
        return float(round(length_mm, 2))

    def skeletonize(self, mask):
        """Упрощенная скелетонизация через морфологию"""
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

    def analyze_image(self, image_path, pixels_per_mm=None, save_dir=TEMP_DIR):
        """
        Полный анализ изображения.
        Если pixels_per_mm не передан, используется default_pixels_per_mm.
        Возвращает словарь с результатами и пути к сохранённым файлам.
        """
        if pixels_per_mm is None:
            pixels_per_mm = self.default_pixels_per_mm

        logger.info(f"Анализ: {os.path.basename(image_path)} (pixels_per_mm={pixels_per_mm})")

        # Инференс
        results = self.model(image_path, conf=0.25, iou=0.45)
        result = results[0]

        # Загрузка изображения
        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]

        # Маски и метрики
        masks_data = []
        total_metrics = {
            'root': {'area_mm2': 0, 'length_mm': 0},
            'stem': {'area_mm2': 0, 'length_mm': 0},
            'leaf': {'area_mm2': 0, 'length_mm': 0}
        }

        # Обработка масок
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()  # (N, H, W)
            class_ids = result.boxes.cls.cpu().numpy() if result.boxes is not None else []

            for i, mask in enumerate(masks):
                # Ресайз маски к оригиналу
                mask_resized = cv2.resize(mask.astype(np.uint8), (img_width, img_height))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)

                # Класс объекта
                cls_id = int(class_ids[i]) if i < len(class_ids) else 0
                cls_name = self.class_names.get(cls_id, 'unknown')

                # Измерения
                area = self.calculate_area_mm2(mask_binary, pixels_per_mm)
                length = self.calculate_length_mm(mask_binary, pixels_per_mm) if cls_name in ['root', 'stem'] else 0

                masks_data.append({
                    'class': cls_name,
                    'class_id': cls_id,
                    'area_mm2': area,
                    'length_mm': length,
                    'confidence': float(result.boxes.conf[i]) if result.boxes is not None else 0.0
                })

                # Суммируем по классам
                if cls_name in total_metrics:
                    total_metrics[cls_name]['area_mm2'] += area
                    if cls_name in ['root', 'stem']:
                        total_metrics[cls_name]['length_mm'] += length

        # Сохранение визуализации
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_') + os.path.splitext(os.path.basename(image_path))[0]

        # Рисуем маски
        if result.masks is not None:
            vis_path = os.path.join(save_dir, f'{timestamp}_mask.jpg')
            result.save(filename=vis_path)
        else:
            vis_path = None

        # Сохраняем метрики
        metrics_path = os.path.join(save_dir, f'{timestamp}_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump({
                'image': os.path.basename(image_path),
                'timestamp': timestamp,
                'pixels_per_mm': pixels_per_mm,
                'objects': masks_data,
                'totals': total_metrics
            }, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

        return {
            'image': image_path,
            'masks': masks_data,
            'totals': total_metrics,
            'visualization': vis_path,
            'metrics_file': metrics_path
        }

# ===================== КАЛИБРОВКА =====================
def calibrate_from_checkerboard(image_path, square_size_mm=10.0, pattern=(9,6)):
    """
    Вычисление pixels_per_mm по одной фотографии шахматки.
    pattern: размер внутренних углов (по умолчанию 9x6, можно передать другие)
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, pattern, None)
    if not ret:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

    # Считаем среднее расстояние между соседними углами
    distances = []
    h, w = pattern
    for i in range(h):
        for j in range(w):
            idx = i * w + j
            if j < w - 1:
                p1 = corners[idx][0]
                p2 = corners[idx+1][0]
                distances.append(np.linalg.norm(p2 - p1))
            if i < h - 1:
                p1 = corners[idx][0]
                p2 = corners[idx + w][0]
                distances.append(np.linalg.norm(p2 - p1))

    if not distances:
        return None
    avg_pixels_per_square = np.mean(distances)
    pixels_per_mm = avg_pixels_per_square / square_size_mm
    return round(pixels_per_mm, 2)


user_data = {}

def get_user_pixels(chat_id):
    return user_data.get(chat_id, {}).get('pixels_per_mm', DEFAULT_PIXELS_PER_MM)

def set_user_pixels(chat_id, value):
    if chat_id not in user_data:
        user_data[chat_id] = {}
    user_data[chat_id]['pixels_per_mm'] = value

def set_calibrating(chat_id, status):
    if chat_id not in user_data:
        user_data[chat_id] = {}
    user_data[chat_id]['calibrating'] = status

def is_calibrating(chat_id):
    return user_data.get(chat_id, {}).get('calibrating', False)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет приветственное сообщение."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Привет, {user.mention_html()}! Я бот для анализа растений (пшеница/руккола).",
        reply_markup=ForceReply(selective=True),
    )
    await update.message.reply_text(
        " Отправь мне фотографию растения, и я определю площадь и длину корней, стеблей и листьев.\n"
        " Для калибровки (определения масштаба) отправь фото шахматной доски (9x6 клеток) после команды /calibrate.\n"
        "Текущий коэффициент: {:.2f} пикс/мм".format(get_user_pixels(update.effective_chat.id))
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет справку."""
    await update.message.reply_text(
        "Команды:\n"
        "/start - приветствие\n"
        "/calibrate - режим калибровки (следующее фото должно быть шахматкой)\n"
        "/status - показать текущий коэффициент калибровки\n"
        "/cancel - отменить режим калибровки\n"
        "Просто отправь фото растения для анализа."
    )

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показывает текущий коэффициент калибровки."""
    ppm = get_user_pixels(update.effective_chat.id)
    await update.message.reply_text(f"Текущий коэффициент: {ppm:.2f} пикс/мм")

async def calibrate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Включает режим калибровки: следующее фото будет использовано для калибровки."""
    chat_id = update.effective_chat.id
    set_calibrating(chat_id, True)
    await update.message.reply_text(
        "Режим калибровки включён. Отправьте фотографию шахматной доски (9x6 внутренних углов).\n"
        "Для отмены используйте /cancel."
    )

async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отменяет режим калибровки."""
    chat_id = update.effective_chat.id
    set_calibrating(chat_id, False)
    await update.message.reply_text("Режим калибровки отменён.")

# ===================== ОБРАБОТЧИК ФОТО =====================
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает полученное фото."""
    chat_id = update.effective_chat.id
    calibrating = is_calibrating(chat_id)

    # Получаем файл фото
    photo_file = await update.message.photo[-1].get_file()
    os.makedirs(TEMP_DIR, exist_ok=True)
    file_path = os.path.join(TEMP_DIR, f"{chat_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    await photo_file.download_to_drive(file_path)

    if calibrating:
        # Режим калибровки
        await update.message.reply_text(" Анализирую шахматку...")
        ppm = calibrate_from_checkerboard(file_path, square_size_mm=CALIB_SQUARE_SIZE_MM)
        if ppm is None:
            await update.message.reply_text(" Не удалось найти шахматную доску (ожидается 9x6 углов). Попробуйте ещё раз или отправьте /cancel.")
        else:
            set_user_pixels(chat_id, ppm)
            await update.message.reply_text(f" Калибровка выполнена! Коэффициент: {ppm:.2f} пикс/мм")
        # Выходим из режима калибровки
        set_calibrating(chat_id, False)
    else:
        # Обычный анализ растения
        await update.message.reply_text(" Анализирую растение... Это может занять несколько секунд.")

        # Получаем коэффициент для этого пользователя
        ppm = get_user_pixels(chat_id)

        # Загружаем модель (она уже загружена в analyzer)
        analyzer = context.bot_data.get('analyzer')
        if analyzer is None:
            await update.message.reply_text(" Ошибка: модель не загружена.")
            return

        try:
            result = analyzer.analyze_image(file_path, pixels_per_mm=ppm, save_dir=TEMP_DIR)

            # Формируем текстовый ответ
            totals = result['totals']
            text = f" **Результаты анализа**\n"
            text += f"Коэффициент: {ppm:.2f} пикс/мм\n\n"
            for part in ['root', 'stem', 'leaf']:
                area = totals[part]['area_mm2']
                length = totals[part]['length_mm']
                if area > 0 or length > 0:
                    text += f"**{part.capitalize()}**\n"
                    text += f"  Площадь: {area} мм²\n"
                    if part in ['root', 'stem']:
                        text += f"  Длина: {length} мм\n"
            if not any(totals[p]['area_mm2'] for p in totals):
                text += "Объекты не обнаружены."

            # Отправляем результат
            await update.message.reply_text(text, parse_mode='Markdown')

            # Если есть визуализация, отправляем её
            if result['visualization'] and os.path.exists(result['visualization']):
                with open(result['visualization'], 'rb') as f:
                    await update.message.reply_photo(photo=f, caption="Визуализация масок")

        except Exception as e:
            logger.exception("Ошибка при анализе")
            await update.message.reply_text(f" Произошла ошибка: {e}")

    # Удаляем временный файл
    if os.path.exists(file_path):
        os.remove(file_path)

# ===================== ОШИБКИ =====================
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Логирует ошибки и уведомляет пользователя."""
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    try:
        if update and update.effective_message:
            await update.effective_message.reply_text("Извините, произошла внутренняя ошибка. Попробуйте позже.")
    except:
        pass

# ===================== ЗАПУСК =====================
def main() -> None:
    """Запускает бота."""
    # Проверка наличия модели
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Модель не найдена по пути {MODEL_PATH}")
        return

    # Инициализация анализатора (загружается один раз)
    analyzer = PlantAnalyzer(MODEL_PATH, default_pixels_per_mm=DEFAULT_PIXELS_PER_MM)

    # Создаём приложение
    application = Application.builder().token(BOT_TOKEN).build()

    # Сохраняем анализатор в bot_data для доступа из обработчиков
    application.bot_data['analyzer'] = analyzer

    # Регистрируем обработчики команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("calibrate", calibrate_command))
    application.add_handler(CommandHandler("cancel", cancel_command))

    # Регистрируем обработчик фото (всех типов)
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Обработчик ошибок
    application.add_error_handler(error_handler)

    # Запускаем бота
    logger.info("Бот запущен...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()