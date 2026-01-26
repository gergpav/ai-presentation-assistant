# app/core/visual_generator.py
"""
Модуль для генерации визуализаций (таблиц и графиков) через matplotlib.
Создает изображения, которые затем вставляются в презентации.
"""
import logging
import io
from pathlib import Path
from typing import Dict, List, Optional
import uuid

import matplotlib
matplotlib.use('Agg')  # Используем backend без GUI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np

logger = logging.getLogger(__name__)

# Настройка matplotlib для поддержки кириллицы
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # Исправляем проблему с минусом


def generate_table_image(content: str, output_dir: Path) -> Optional[str]:
    """
    Генерирует изображение таблицы из текстового контента.
    
    Args:
        content: Текст в формате "Столбец1 | Столбец2 | ..."
        output_dir: Директория для сохранения изображения
    
    Returns:
        Путь к сохраненному изображению или None при ошибке
    """
    try:
        # Парсим таблицу из текста
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return None
        
        table_rows = []
        for line in lines:
            # Убираем маркеры списка и markdown форматирование
            clean_line = line.strip()
            if clean_line.startswith('•') or clean_line.startswith('-') or clean_line.startswith('*'):
                clean_line = clean_line[1:].strip()
            clean_line = clean_line.replace('```', '').replace('`', '').replace('**', '').replace('*', '')
            
            # Проверяем наличие разделителя |
            if '|' in clean_line:
                cells = [cell.strip() for cell in clean_line.split('|')]
                cells = [cell for cell in cells if cell]  # Фильтруем пустые
                if len(cells) >= 2:
                    table_rows.append(cells)
        
        if not table_rows or len(table_rows) < 2:
            logger.warning(f"Не удалось распарсить таблицу (найдено строк: {len(table_rows)})")
            return None
        
        # Определяем количество столбцов
        num_cols = max(len(row) for row in table_rows)
        num_rows = len(table_rows)
        
        # Создаем фигуру
        fig, ax = plt.subplots(figsize=(12, max(6, num_rows * 0.8)))
        ax.axis('tight')
        ax.axis('off')
        
        # Подготовка данных для таблицы
        table_data = []
        for row in table_rows:
            # Дополняем строку до нужного количества столбцов
            padded_row = row + [''] * (num_cols - len(row))
            table_data.append(padded_row)
        
        # Создаем таблицу
        table = ax.table(
            cellText=table_data,
            cellLoc='left',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        # Стилизуем таблицу
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Стили для заголовка (первая строка)
        for i in range(num_cols):
            cell = table[(0, i)]
            cell.set_facecolor('#4472C4')  # Синий
            cell.set_text_props(weight='bold', color='white')
            cell.set_height(0.15)
        
        # Стили для остальных строк
        for row_idx in range(1, num_rows):
            for col_idx in range(num_cols):
                cell = table[(row_idx, col_idx)]
                cell.set_facecolor('#FFFFFF')
                cell.set_text_props(color='black')
                cell.set_height(0.12)
        
        # Сохраняем изображение
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"table_{uuid.uuid4().hex}.png"
        output_path = output_dir / filename
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Таблица сохранена: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Ошибка при генерации таблицы: {e}", exc_info=True)
        plt.close('all')
        return None


def generate_chart_image(content: str, output_dir: Path) -> Optional[str]:
    """
    Генерирует изображение графика из текстового контента.
    
    Args:
        content: Текст в формате "Название: Значение"
        output_dir: Директория для сохранения изображения
    
    Returns:
        Путь к сохраненному изображению или None при ошибке
    """
    try:
        # Парсим данные графика
        chart_data = {}
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        for line in lines:
            # Убираем маркеры списка и markdown форматирование
            clean_line = line.strip()
            if clean_line.startswith('•') or clean_line.startswith('-') or clean_line.startswith('*'):
                clean_line = clean_line[1:].strip()
            clean_line = clean_line.replace('```', '').replace('`', '').replace('**', '').replace('*', '')
            
            # Проверяем наличие двоеточия
            if ':' not in clean_line:
                continue
            
            parts = clean_line.split(':', 1)
            if len(parts) == 2:
                name = parts[0].strip()
                value_str = parts[1].strip()
                
                # Пытаемся извлечь числовое значение
                try:
                    cleaned_value = value_str.replace('$', '').replace('₽', '').replace('€', '').replace(',', '').replace(' ', '')
                    numeric_str = ''.join(c for c in cleaned_value if c.isdigit() or c == '.' or c == '-')
                    if numeric_str:
                        value = float(numeric_str)
                        chart_data[name] = value
                except (ValueError, AttributeError):
                    logger.debug(f"Не удалось извлечь число из '{value_str}', пропускаем")
                    continue
        
        if not chart_data:
            logger.warning(f"Не удалось распарсить данные графика (найдено записей: {len(chart_data)})")
            return None
        
        # Создаем график
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Подготовка данных
        names = list(chart_data.keys())
        values = list(chart_data.values())
        
        # Создаем столбчатую диаграмму
        bars = ax.bar(names, values, color='#4472C4', edgecolor='#2E5C8A', linewidth=1.5)
        
        # Добавляем значения на столбцы
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Настройка осей
        ax.set_ylabel('Значение', fontsize=12, fontweight='bold')
        ax.set_xlabel('Показатель', fontsize=12, fontweight='bold')
        ax.set_title('График показателей', fontsize=14, fontweight='bold', pad=20)
        
        # Поворачиваем подписи на оси X для лучшей читаемости
        plt.xticks(rotation=45, ha='right')
        
        # Сетка
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Сохраняем изображение
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"chart_{uuid.uuid4().hex}.png"
        output_path = output_dir / filename
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"График сохранен: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Ошибка при генерации графика: {e}", exc_info=True)
        plt.close('all')
        return None
