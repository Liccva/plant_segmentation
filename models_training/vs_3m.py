import torch
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from datetime import datetime
import pandas as pd
from pathlib import Path

if __name__ == '__main__':

    print("=" * 80)
    print(" СРАВНЕНИЕ ТРЕХ МОДЕЛЕЙ НА ТЕСТОВОЙ И ВАЛИДАЦИОННОЙ ВЫБОРКАХ")
    print("=" * 80)

    # ===================== ПРОВЕРКА GPU =====================
    print("\n ПРОВЕРКА СИСТЕМЫ")
    print("=" * 80)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} ГБ")
        device = 0
    else:
        print(" CUDA не найдена, использую CPU")
        device = 'cpu'

    # ===================== ПУТЬ К ДАТАСЕТУ =====================
    DATA_YAML = "C:/AI_hackathon/data_3x/data.yaml"

    if not os.path.exists(DATA_YAML):
        print(f" Файл датасета не найден: {DATA_YAML}")
        exit(1)

    print(f" Использую датасет: {DATA_YAML}")

    # ===================== ПУТИ К МОДЕЛЯМ =====================
    print("\n ЗАГРУЗКА МОДЕЛЕЙ")
    print("=" * 80)

    models = [
        {
            'name': 'baseline_s_1024',
            'path': r"runs/segment/runs/segment/baseline_s_1024_20260309_012026/weights/best.pt",
            'color': 'blue',
            'marker': 'o'
        },
        {
            'name': 'highres_s_1280_b4',
            'path': r"runs/segment/highres_s_1280_b4_20260309_0745432/weights/best.pt",
            'color': 'red',
            'marker': 's'
        },
        {
            'name': 'strong_aug_s_1152',
            'path': r"runs/segment/strong_aug_s_1152_20260309_0951462/weights/best.pt",
            'color': 'green',
            'marker': '^'
        }
    ]

    # Проверяем существование моделей
    loaded_models = []
    for model_info in models:
        if os.path.exists(model_info['path']):
            print(f" {model_info['name']}: найдена")
            model_info['model'] = YOLO(model_info['path'])
            loaded_models.append(model_info)
        else:
            print(f" {model_info['name']}: НЕ найдена по пути {model_info['path']}")

    if not loaded_models:
        print(" Нет доступных моделей для сравнения!")
        exit(1)


    # ===================== ФУНКЦИЯ ДЛЯ СБОРА МЕТРИК =====================
    def collect_metrics(models_list, data_yaml, split='val'):
        """Собирает метрики для всех моделей на указанной выборке"""
        results = []

        for model_info in models_list:
            print(f"\n Оценка {model_info['name']} на {split}...")

            # Валидация модели - передаем путь к data.yaml
            metrics = model_info['model'].val(
                data=data_yaml,
                split=split  # указываем, какую выборку использовать
            )

            # Собираем метрики
            model_result = {
                'model_name': model_info['name'],
                'split': split,
                'box_mAP50': float(metrics.box.map50),
                'box_mAP50-95': float(metrics.box.map),
                'mask_mAP50': float(metrics.seg.map50),
                'mask_mAP50-95': float(metrics.seg.map),
                'precision': float(metrics.box.mp),
                'recall': float(metrics.box.mr),
                'per_class': {}
            }

            # Метрики по классам
            class_names = ['leaf', 'root', 'stem']
            for i, class_name in enumerate(class_names):
                model_result['per_class'][class_name] = {
                    'box_mAP50': float(metrics.box.maps[i] * 100) if i < len(metrics.box.maps) else 0,
                    'mask_mAP50': float(metrics.seg.maps[i] * 100) if i < len(metrics.seg.maps) else 0,
                    'precision': float(metrics.box.p[i]) if i < len(metrics.box.p) else 0,
                    'recall': float(metrics.box.r[i]) if i < len(metrics.box.r) else 0
                }

            results.append(model_result)

            # Вывод в консоль
            print(f"   all mAP50: {model_result['box_mAP50'] * 100:.1f}%")
            for class_name in class_names:
                box_map = model_result['per_class'][class_name]['box_mAP50']
                print(f"   {class_name}: {box_map:.1f}%")

        return results


    # ===================== СБОР МЕТРИК =====================
    print("\n" + "=" * 80)
    print(" СБОР МЕТРИК НА ВАЛИДАЦИОННОЙ ВЫБОРКЕ")
    print("=" * 80)

    val_results = collect_metrics(loaded_models, DATA_YAML, split='val')

    print("\n" + "=" * 80)
    print(" СБОР МЕТРИК НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("=" * 80)

    test_results = collect_metrics(loaded_models, DATA_YAML, split='test')

    # ===================== СОЗДАНИЕ ТАБЛИЦ СРАВНЕНИЯ =====================
    print("\n" + "=" * 80)
    print(" ИТОГОВЫЕ ТАБЛИЦЫ СРАВНЕНИЯ")
    print("=" * 80)


    # Функция для создания таблицы
    def print_comparison_table(results, split_name):
        print(f"\n{'=' * 100}")
        print(f" РЕЗУЛЬТАТЫ НА {split_name.upper()} ВЫБОРКЕ")
        print(f"{'=' * 100}")

        # Заголовок таблицы
        print(f"{'Модель':<20} {'Class':<8} {'Box mAP50':<10} {'Mask mAP50':<10} {'Precision':<10} {'Recall':<10}")
        print("-" * 70)

        for res in results:
            model_name = res['model_name']
            first_row = True

            for class_name in ['leaf', 'root', 'stem']:
                class_data = res['per_class'][class_name]

                if first_row:
                    print(
                        f"{model_name:<20} {class_name:<8} {class_data['box_mAP50']:<10.1f} {class_data['mask_mAP50']:<10.1f} {class_data['precision'] * 100:<10.1f} {class_data['recall'] * 100:<10.1f}")
                    first_row = False
                else:
                    print(
                        f"{'':<20} {class_name:<8} {class_data['box_mAP50']:<10.1f} {class_data['mask_mAP50']:<10.1f} {class_data['precision'] * 100:<10.1f} {class_data['recall'] * 100:<10.1f}")

            # Общие метрики
            print(
                f"{'':<20} {'ALL':<8} {res['box_mAP50'] * 100:<10.1f} {res['mask_mAP50'] * 100:<10.1f} {res['precision'] * 100:<10.1f} {res['recall'] * 100:<10.1f}")
            print("-" * 70)


    # Печатаем таблицы
    print_comparison_table(val_results, 'валидационной')
    print_comparison_table(test_results, 'тестовой')

    # ===================== ВИЗУАЛИЗАЦИЯ =====================
    print("\n СОЗДАНИЕ ГРАФИКОВ СРАВНЕНИЯ")
    print("=" * 80)

    # Создаем папку для результатов
    comparison_dir = "C:/AI_hackathon/model_comparison"
    os.makedirs(comparison_dir, exist_ok=True)

    # 1. Сравнение по корням (главная метрика)
    plt.figure(figsize=(12, 6))

    x = range(len(loaded_models))
    root_val = [r['per_class']['root']['box_mAP50'] for r in val_results]
    root_test = [r['per_class']['root']['box_mAP50'] for r in test_results]

    plt.subplot(1, 2, 1)
    bars = plt.bar(x, root_val, color=[m['color'] for m in loaded_models])
    plt.xticks(x, [m['name'] for m in loaded_models], rotation=45, ha='right')
    plt.ylabel('root mAP50 (%)')
    plt.title('Сравнение по корням (валидация)')
    plt.grid(True, alpha=0.3)

    # Добавляем значения на столбцы
    for bar, val in zip(bars, root_val):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{val:.1f}%',
                 ha='center', va='bottom')

    plt.subplot(1, 2, 2)
    bars = plt.bar(x, root_test, color=[m['color'] for m in loaded_models])
    plt.xticks(x, [m['name'] for m in loaded_models], rotation=45, ha='right')
    plt.ylabel('root mAP50 (%)')
    plt.title('Сравнение по корням (тест)')
    plt.grid(True, alpha=0.3)

    for bar, val in zip(bars, root_test):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{val:.1f}%',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'comparison_root.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # 2. Радарная диаграмма для всех классов
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(projection='polar'))

    for idx, (results, split_name, ax) in enumerate(zip([val_results, test_results],
                                                        ['Валидация', 'Тест'],
                                                        axes)):

        angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
        angles += angles[:1]  # замыкаем круг

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        for i, res in enumerate(results):
            values = [
                res['per_class']['leaf']['box_mAP50'],
                res['per_class']['root']['box_mAP50'],
                res['per_class']['stem']['box_mAP50']
            ]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2, label=res['model_name'],
                    color=models[i]['color'], markersize=8, marker=models[i]['marker'])
            ax.fill(angles, values, alpha=0.1, color=models[i]['color'])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['Leaf', 'Root', 'Stem'])
        ax.set_ylim(0, 100)
        ax.set_title(f'{split_name} - Сравнение по классам', size=12)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'comparison_radar.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # 3. Сравнение Precision/Recall для корней
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    x = range(len(loaded_models))
    precisions = [r['per_class']['root']['precision'] * 100 for r in val_results]
    recalls = [r['per_class']['root']['recall'] * 100 for r in val_results]

    plt.plot(x, precisions, 'o-', label='Precision', linewidth=2, markersize=10, color='blue')
    plt.plot(x, recalls, 's-', label='Recall', linewidth=2, markersize=10, color='red')
    plt.xticks(x, [m['name'] for m in loaded_models], rotation=45, ha='right')
    plt.ylabel('Процент (%)')
    plt.title('Precision/Recall для корней (валидация)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    precisions_test = [r['per_class']['root']['precision'] * 100 for r in test_results]
    recalls_test = [r['per_class']['root']['recall'] * 100 for r in test_results]

    plt.plot(x, precisions_test, 'o-', label='Precision', linewidth=2, markersize=10, color='blue')
    plt.plot(x, recalls_test, 's-', label='Recall', linewidth=2, markersize=10, color='red')
    plt.xticks(x, [m['name'] for m in loaded_models], rotation=45, ha='right')
    plt.ylabel('Процент (%)')
    plt.title('Precision/Recall для корней (тест)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'comparison_pr.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # ===================== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ =====================
    print("\n СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 80)

    # Сохраняем в JSON
    all_results = {
        'validation': val_results,
        'test': test_results,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    json_path = os.path.join(comparison_dir, 'comparison_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f" JSON сохранен: {json_path}")


    # Сохраняем в CSV для Excel
    def results_to_csv(results, filename):
        rows = []
        for res in results:
            for class_name in ['leaf', 'root', 'stem']:
                row = {
                    'model': res['model_name'],
                    'split': res['split'],
                    'class': class_name,
                    'box_mAP50': res['per_class'][class_name]['box_mAP50'],
                    'mask_mAP50': res['per_class'][class_name]['mask_mAP50'],
                    'precision': res['per_class'][class_name]['precision'] * 100,
                    'recall': res['per_class'][class_name]['recall'] * 100
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(comparison_dir, filename), index=False, encoding='utf-8-sig')
        print(f" CSV сохранен: {os.path.join(comparison_dir, filename)}")


    results_to_csv(val_results, 'validation_results.csv')
    results_to_csv(test_results, 'test_results.csv')

    # ===================== ФИНАЛЬНЫЙ ВЫВОД =====================
    print("\n" + "=" * 80)
    print(" ИТОГОВОЕ СРАВНЕНИЕ ПО КОРНЯМ")
    print("=" * 80)

    # Находим лучшую модель для корней
    best_root_val = max(val_results, key=lambda x: x['per_class']['root']['box_mAP50'])
    best_root_test = max(test_results, key=lambda x: x['per_class']['root']['box_mAP50'])

    print(f"\n Лучшая модель на ВАЛИДАЦИИ для корней:")
    print(f"   {best_root_val['model_name']} - root mAP50 = {best_root_val['per_class']['root']['box_mAP50']:.1f}%")
    print(f"   leaf mAP50 = {best_root_val['per_class']['leaf']['box_mAP50']:.1f}%")
    print(f"   stem mAP50 = {best_root_val['per_class']['stem']['box_mAP50']:.1f}%")

    print(f"\n Лучшая модель на ТЕСТЕ для корней:")
    print(f"   {best_root_test['model_name']} - root mAP50 = {best_root_test['per_class']['root']['box_mAP50']:.1f}%")

    # Сохраняем информацию о лучшей модели
    best_info = {
        'best_val': {
            'model': best_root_val['model_name'],
            'root_mAP50': best_root_val['per_class']['root']['box_mAP50']
        },
        'best_test': {
            'model': best_root_test['model_name'],
            'root_mAP50': best_root_test['per_class']['root']['box_mAP50']
        }
    }

    best_info_path = os.path.join(comparison_dir, 'best_model_info.json')
    with open(best_info_path, 'w', encoding='utf-8') as f:
        json.dump(best_info, f, indent=2, ensure_ascii=False)

    print(f"\n Все результаты сохранены в: {comparison_dir}")
    print("=" * 80)