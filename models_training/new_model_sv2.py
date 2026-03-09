import torch
import os
from ultralytics import YOLO
from datetime import datetime
import json
import time

if __name__ == '__main__':




    print("\n ПРОВЕРКА СИСТЕМЫ")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} ГБ")
        torch.cuda.empty_cache()
        device = 0
    else:
        print(" CUDA не найдена!")
        exit(1)


    BASE_DIR = "C:/AI_hackathon/runs/segment"
    DATASET_PATH = "C:/AI_hackathon/data_3x/data.yaml"



    # ===================== ОПРЕДЕЛЕНИЕ ОСТАВШИХСЯ ВАРИАНТОВ =====================
    experiments = [
        {
            'name': 'baseline_s_1024',
            'model': 'yolo26s-seg.pt',
            'imgsz': 1024,
            'batch': 8,
            'cls': 2.0,
            'box': 8.5,
            'lr0': 0.0004,
            'description': 'Базовая s-модель с 1024px',
        },
        {
            'name': 'highres_s_1280_b4',
            'model': 'yolo26s-seg.pt',
            'imgsz': 1280,
            'batch': 4,
            'cls': 2.2,
            'box': 9.0,
            'lr0': 0.0003,
            'description': 'Высокое разрешение 1280px, batch=4',
        },
        {
            'name': 'strong_aug_s_1152',
            'model': 'yolo26s-seg.pt',
            'imgsz': 1152,
            'batch': 6,
            'cls': 2.5,
            'box': 9.5,
            'lr0': 0.0005,
            'description': 'Усиленная аугментация, 1152px',
            'hsv_h': 0.04,
            'hsv_s': 0.7,
            'hsv_v': 0.5,
            'degrees': 10.0,
            'scale': 0.5,
            'fliplr': 0.5,
            'mosaic': 0.5,
            'mixup': 0.3,
        }
    ]


    # ===================== ФУНКЦИЯ ДЛЯ ОБУЧЕНИЯ =====================
    def train_experiment(exp_config):
        print("\n" + "=" * 70)
        print(f" ЭКСПЕРИМЕНТ: {exp_config['name']}")
        print(f" Описание: {exp_config['description']}")
        print("=" * 70)

        # Создаем уникальную папку с датой
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{exp_config['name']}_{current_time}"
        experiment_dir = os.path.join(BASE_DIR, experiment_name)
        os.makedirs(experiment_dir, exist_ok=False)

        print(f" Папка результатов: {experiment_dir}")

        # Сохраняем конфигурацию
        config_path = os.path.join(experiment_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(exp_config, f, indent=2)

        # Загружаем модель
        print(f"\n Загрузка модели {exp_config['model']}...")
        model = YOLO(exp_config['model'])

        # Параметры аугментации
        aug_params = {
            'hsv_h': exp_config.get('hsv_h', 0.02),
            'hsv_s': exp_config.get('hsv_s', 0.4),
            'hsv_v': exp_config.get('hsv_v', 0.3),
            'degrees': exp_config.get('degrees', 5.0),
            'translate': exp_config.get('translate', 0.1),
            'scale': exp_config.get('scale', 0.3),
            'fliplr': exp_config.get('fliplr', 0.2),
            'mosaic': exp_config.get('mosaic', 0.3),
            'mixup': exp_config.get('mixup', 0.2),
        }

        print(f"\n Параметры:")
        print(f"   imgsz: {exp_config['imgsz']}")
        print(f"   batch: {exp_config['batch']}")
        print(f"   cls: {exp_config['cls']}")
        print(f"   box: {exp_config['box']}")
        print(f"   lr0: {exp_config['lr0']}")

        # Засекаем время
        start_time = time.time()

        # Запуск обучения
        results = model.train(
            data=DATASET_PATH,
            epochs=150,
            imgsz=exp_config['imgsz'],
            batch=exp_config['batch'],
            device=0,
            patience=40,

            # Веса
            cls=exp_config['cls'],
            box=exp_config['box'],

            # Аугментация
            **aug_params,

            # Оптимизация
            optimizer='AdamW',
            lr0=exp_config['lr0'],
            warmup_epochs=5,

            # Сохранение - исправлено!
            project=BASE_DIR,
            name=experiment_name,
            exist_ok=False,
            plots=True,
            verbose=True,
        )

        # Время обучения
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)

        print(f"\n Эксперимент завершен за {hours}ч {minutes}м")
        print(f" Результаты: {experiment_dir}")

        # Валидация лучшей модели
        best_model_path = os.path.join(experiment_dir, 'weights', 'best.pt')
        print(f"\n Загрузка лучшей модели из: {best_model_path}")

        if os.path.exists(best_model_path):
            best_model = YOLO(best_model_path)
            metrics = best_model.val()

            # Сохраняем метрики
            metrics_dict = {
                'experiment': exp_config['name'],
                'box_mAP50': float(metrics.box.map50),
                'box_mAP50-95': float(metrics.box.map),
                'mask_mAP50': float(metrics.seg.map50),
                'mask_mAP50-95': float(metrics.seg.map),
                'per_class': {}
            }

            class_names = ['leaf', 'root', 'stem']
            for i, name in enumerate(class_names):
                metrics_dict['per_class'][name] = {
                    'box_mAP50': float(metrics.box.maps[i] * 100) if i < len(metrics.box.maps) else 0,
                    'mask_mAP50': float(metrics.seg.maps[i] * 100) if i < len(metrics.seg.maps) else 0
                }

            metrics_path = os.path.join(experiment_dir, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics_dict, f, indent=2)

            # Вывод результатов
            print(f"\n РЕЗУЛЬТАТЫ {exp_config['name']}:")
            for name in class_names:
                box_map = metrics_dict['per_class'][name]['box_mAP50']
                mask_map = metrics_dict['per_class'][name]['mask_mAP50']
                print(f"   {name}: box mAP50 = {box_map:.1f}%, mask mAP50 = {mask_map:.1f}%")

            return metrics_dict, experiment_dir
        else:
            print(f" Модель не найдена по пути: {best_model_path}")
            return None, experiment_dir


    # ===================== ЗАПУСК ОСТАВШИХСЯ ЭКСПЕРИМЕНТОВ =====================
    results_list = []
    experiment_dirs = []

    # Добавляем информацию о первом эксперименте вручную
    first_exp_results = {
        'experiment': 'baseline_s_1024',
        'per_class': {
            'leaf': {'box_mAP50': 97.0},
            'root': {'box_mAP50': 35.2},
            'stem': {'box_mAP50': 83.9}
        }
    }


    print(f"\n{'#' * 70}")
    print(f"# ЗАПУСК ЭКСПЕРИМЕНТОВ 1 И 3")
    print(f"{'#' * 70}")

    for i, exp in enumerate(experiments, 1):  # начинаем с 1
        print(f"\n{'#' * 70}")
        print(f"# ЭКСПЕРИМЕНТ {i}/3: {exp['name']}")
        print(f"{'#' * 70}")

        # Очищаем кэш перед каждым экспериментом
        torch.cuda.empty_cache()

        # Запускаем обучение
        metrics, exp_dir = train_experiment(exp)
        if metrics:
            results_list.append(metrics)
        experiment_dirs.append(exp_dir)

        # Пауза между экспериментами
        if i < 3:
            print(f"\n Пауза 30 секунд перед следующим экспериментом...")
            time.sleep(30)

    # ===================== СРАВНЕНИЕ ВСЕХ РЕЗУЛЬТАТОВ =====================
    print("\n" + "=" * 70)
    print(" СРАВНЕНИЕ ВСЕХ ЭКСПЕРИМЕНТОВ")
    print("=" * 70)

    print(f"\n{'Эксперимент':<30} {'leaf mAP50':<12} {'root mAP50':<12} {'stem mAP50':<12}")
    print("-" * 66)

    for i, exp_dir in enumerate(experiment_dirs):
        # Пытаемся загрузить метрики из JSON
        metrics_path = os.path.join(exp_dir, 'metrics.json') if i > 0 else None

        # Загружаем из JSON
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            name = metrics['experiment']
            leaf = metrics['per_class']['leaf']['box_mAP50']
            root = metrics['per_class']['root']['box_mAP50']
            stem = metrics['per_class']['stem']['box_mAP50']
            print(f"{name:<30} {leaf:<12.1f} {root:<12.1f} {stem:<12.1f}")
        else:
            print(f"{exp_dir:<30} - файл метрик не найден")

    print("-" * 66)
    print("\n ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
    print(" Результаты сохранены в:")
    for exp_dir in experiment_dirs:
        print(f"   • {exp_dir}")