# app/core/pdf_builder.py
import io
import logging
import os
from typing import List
from pathlib import Path

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.units import cm
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from app.utils.helpers import SlideExport

logger = logging.getLogger(__name__)

# Регистрируем TTF-шрифты с поддержкой Unicode/кириллицы
# Используем DejaVu Sans - свободный шрифт с отличной поддержкой Unicode
_FONTS_REGISTERED = False

def _register_fonts():
    """Регистрирует TTF-шрифты с поддержкой кириллицы"""
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return
    
    # Список путей для поиска шрифтов (системные пути Linux/Windows)
    font_paths = [
        # Linux системные пути
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        # Windows системные пути
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/timesbd.ttf",
        # macOS системные пути
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        # Альтернативные пути
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    
    # Имена шрифтов для использования в reportlab
    regular_font_name = "DejaVuSans"
    bold_font_name = "DejaVuSans-Bold"
    
    # Пытаемся найти и зарегистрировать DejaVu Sans
    regular_font_path = None
    bold_font_path = None
    
    for path in font_paths:
        if os.path.exists(path):
            if "Bold" in path or "bd" in path.lower() or "bold" in path.lower():
                if not bold_font_path:
                    bold_font_path = path
            else:
                if not regular_font_path and ("DejaVu" in path or "Arial" in path or "Liberation" in path):
                    regular_font_path = path
    
    # Если не нашли DejaVu, пытаемся использовать Arial или Liberation Sans
    if not regular_font_path:
        for path in font_paths:
            if os.path.exists(path) and ("arial" in path.lower() or "liberation" in path.lower()):
                regular_font_path = path
                if "bold" not in path.lower() and "bd" not in path.lower():
                    break
    
    # Регистрируем найденные шрифты
    try:
        if regular_font_path:
            pdfmetrics.registerFont(TTFont(regular_font_name, regular_font_path))
            logger.info(f"Зарегистрирован шрифт для кириллицы: {regular_font_name} из {regular_font_path}")
        else:
            # Если не нашли системные шрифты, используем встроенные шрифты reportlab
            # Но они не поддерживают кириллицу, поэтому будет предупреждение
            logger.warning("Не найдены TTF-шрифты с поддержкой кириллицы. Кириллица может отображаться некорректно.")
            logger.warning("Рекомендуется установить DejaVu Sans: apt-get install fonts-dejavu")
            regular_font_name = "Helvetica"  # Fallback
        
        if bold_font_path:
            pdfmetrics.registerFont(TTFont(bold_font_name, bold_font_path))
            logger.info(f"Зарегистрирован жирный шрифт для кириллицы: {bold_font_name} из {bold_font_path}")
        else:
            bold_font_name = regular_font_name  # Используем обычный шрифт как fallback
    except Exception as e:
        logger.error(f"Ошибка при регистрации шрифтов: {e}")
        regular_font_name = "Helvetica"
        bold_font_name = "Helvetica-Bold"
    
    # Сохраняем имена шрифтов для использования
    _register_fonts.regular_font = regular_font_name
    _register_fonts.bold_font = bold_font_name
    _FONTS_REGISTERED = True

# Инициализируем шрифты при импорте модуля
_register_fonts()


def slides_to_pdf_bytes(slides: List[SlideExport], audience: str) -> bytes:
    """
    Генерация PDF-презентации напрямую из структуры слайдов.
    Каждый слайд = отдельная страница PDF.
    Поддерживает разные типы визуализации и макеты.
    """
    buffer = io.BytesIO()

    # Горизонтальная страница формата A4 — условно "слайд"
    page_size = landscape(A4)
    c = canvas.Canvas(buffer, pagesize=page_size)
    width, height = page_size

    # Стили для текста
    styles = getSampleStyleSheet()
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=14,
        leading=18,
        leftIndent=0,
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=normal_style,
        leftIndent=20,
        bulletIndent=10,
    )

    # Приводим audience в человекочитаемый вид (русские названия)
    audience_map = {
        "AudienceType.investors": "инвесторы",
        "AudienceType.experts": "эксперты",
        "AudienceType.management": "руководство",
        "investors": "инвесторы",
        "experts": "эксперты",
        "management": "руководство",
    }
    audience_display = audience_map.get(str(audience), str(audience))

    for idx, slide in enumerate(slides, start=1):
        layout_type = getattr(slide, "layout", "title_and_content")
        visual_type_raw = getattr(slide, "visual_type", "text")
        if hasattr(visual_type_raw, "value"):
            visual_type = visual_type_raw.value
        else:
            visual_type = str(visual_type_raw)
        visual_type = visual_type.lower()
        if "." in visual_type:
            visual_type = visual_type.split(".")[-1]
        
        # --- Заголовок ---
        title = slide.title or "Слайд"
        
        # Для титульного слайда парсим заголовок и подзаголовок из контента
        subtitle_text = None
        if layout_type == "title" and slide.content:
            content_lines = [line.strip() for line in slide.content.split('\n') if line.strip()]
            for line in content_lines:
                clean_line = line.lstrip("•-* ").strip()
                if clean_line.startswith("ЗАГОЛОВОК:") or clean_line.startswith("Заголовок:"):
                    title = clean_line.split(":", 1)[1].strip() if ":" in clean_line else clean_line
                elif clean_line.startswith("ПОДЗАГОЛОВОК:") or clean_line.startswith("Подзаголовок:"):
                    subtitle_text = clean_line.split(":", 1)[1].strip() if ":" in clean_line else clean_line
            
            # Если не нашли структурированный формат, используем первую строку как заголовок
            if title == slide.title and content_lines:
                title = content_lines[0].lstrip("•-* ").strip()
                if len(content_lines) > 1:
                    subtitle_text = content_lines[1].lstrip("•-* ").strip()
        
        # Используем Paragraph с TTF-шрифтом для поддержки Unicode/кириллицы
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Normal'],
            fontSize=24,
            fontName=_register_fonts.bold_font,  # Используем зарегистрированный TTF-шрифт
            leading=28,
        )
        title_para = Paragraph(title, title_style)
        # Увеличиваем ширину заголовка, чтобы длинные названия не выходили за рамки
        title_para.wrapOn(c, width - 1 * cm, height)  # Почти на всю ширину
        # Позиционирование заголовка: титульный — ближе к центру, остальные — чуть ниже верхней части
        if layout_type == "title":
            title_y = (height - title_para.height) * 0.55
        else:
            # Поднимаем заголовок выше примерно на 2 см
            title_y = height - 0.8 * cm - title_para.height
        title_para.drawOn(c, 0.5 * cm, title_y)

        # --- Подзаголовок ---
        if layout_type == "title" and subtitle_text:
            # Для титульного слайда используем подзаголовок из контента
            subtitle_style = ParagraphStyle(
                'SubtitleStyle',
                parent=styles['Normal'],
                fontSize=16,
                fontName=_register_fonts.regular_font,
                leading=20,
            )
            subtitle_para = Paragraph(subtitle_text.lstrip("•-* ").strip(), subtitle_style)
            subtitle_para.wrapOn(c, width - 1 * cm, height)
            subtitle_y = title_y - 1.2 * cm - subtitle_para.height
            subtitle_para.drawOn(c, 0.5 * cm, subtitle_y)
        elif layout_type != "title":
            # Для обычных слайдов показываем аудиторию и номер
            subtitle = f"Аудитория: {audience_display} • Слайд {idx} из {len(slides)}"
            subtitle_style = ParagraphStyle(
                'SubtitleStyle',
                parent=styles['Normal'],
                fontSize=10,
                fontName=_register_fonts.regular_font,  # Используем зарегистрированный TTF-шрифт
                leading=12,
            )
            subtitle_para = Paragraph(subtitle, subtitle_style)
            subtitle_para.wrapOn(c, width - 4 * cm, height)
            subtitle_y = height - 3 * cm - subtitle_para.height
            subtitle_para.drawOn(c, 2 * cm, subtitle_y)

        # Для титульного слайда или слайда только с заголовком - пропускаем контент
        # (не добавляем буллеты или другой контент)
        if layout_type == "title" or layout_type == "title_only":
            c.showPage()
            continue

        # --- Обработка контента в зависимости от типа визуализации ---
        # Текст возвращаем к прежнему положению; визуализации опускаем на ~5 см
        text_x = 2 * cm
        text_y = height - 5 * cm
        text_width = width - 4 * cm
        text_max_height = height - text_y - 1 * cm

        # Смещаем визуализации правее на 2 см
        visual_x = 3 * cm
        visual_width = text_width - 2 * cm
        # Располагаем визуализации под строкой аудитории и ближе к середине слайда
        visual_top_limit = subtitle_y - 0.8 * cm
        visual_bottom_limit = 2 * cm
        available_height = max(visual_top_limit - visual_bottom_limit, 2 * cm)
        # Фиксируем максимальную высоту и центрируем по доступной области
        visual_max_height = min(10 * cm, available_height)
        visual_y = visual_bottom_limit + (available_height + visual_max_height) / 2
        # Поднимаем визуализации ещё на 2 см, но не выше верхнего лимита
        visual_y = min(visual_top_limit, visual_y + 2 * cm)

        # Получаем список изображений
        images = getattr(slide, "images", []) or []

        if visual_type == "table":
            # Только изображение таблицы, без текстового описания
            if images:
                _draw_image(c, images[0], visual_x, visual_y, visual_width, visual_max_height)
            # Если нет изображения — ничего не рисуем (описание убрано)
        elif visual_type == "chart":
            # Только изображение графика, без текстового описания
            if images:
                _draw_image(c, images[0], visual_x, visual_y, visual_width, visual_max_height)
            # Если нет изображения — ничего не рисуем (описание убрано)
        elif visual_type == "image":
            # Только изображение, без текстового описания
            if images:
                _draw_image(c, images[0], visual_x, visual_y, visual_width, visual_max_height)
            # Если нет изображения — ничего не рисуем (описание убрано)
        else:  # text
            _draw_text_content(c, slide.content, text_x, text_y, text_width, text_max_height)

        # --- Дополнительные картинки (если есть и тип не image/table/chart) ---
        if images and visual_type not in ["image", "table", "chart"]:
            img_y = 2.5 * cm
            img_x = 2 * cm
            img_max_width = 8 * cm
            img_max_height = 5 * cm

            for img_path in images[:2]:  # максимум 2 картинки на слайд
                try:
                    c.drawImage(
                        img_path,
                        img_x,
                        img_y,
                        width=img_max_width,
                        height=img_max_height,
                        preserveAspectRatio=True,
                        anchor="sw",
                    )
                    img_x += img_max_width + 1 * cm
                except Exception as e:
                    logger.warning(f"Не удалось добавить изображение '{img_path}' на слайд {idx}: {e}")

        # --- Завершаем страницу ---
        c.showPage()

    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def _draw_text_content(c: canvas.Canvas, content: str, x: float, y: float, width: float, max_height: float):
    """Рисует текстовый контент на PDF с правильным форматированием и поддержкой Unicode/кириллицы"""
    if not content or not content.strip():
        return
    
    styles = getSampleStyleSheet()
    current_y = y
    line_height = 0.9 * cm
    
    for raw_line in content.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        # Убираем markdown форматирование
        line = line.replace("**", "").replace("*", "").replace("__", "").replace("_", "")
        
        # Убираем маркеры списка если есть (они будут добавлены автоматически)
        if line.startswith("•"):
            line = line[1:].strip()
        elif line.startswith("-"):
            line = line[1:].strip()
        
        # Определяем жирный текст (если вся строка заглавными или содержит ключевые слова)
        is_bold = line.isupper() or any(kw in line.lower() for kw in ['важно', 'ключевой', 'основной'])
        
        # Используем Paragraph с TTF-шрифтом для поддержки Unicode/кириллицы
        if is_bold:
            para_style = ParagraphStyle(
                'BoldText',
                parent=styles['Normal'],
                fontSize=14,
                fontName=_register_fonts.bold_font,  # Используем зарегистрированный TTF-шрифт
                leading=18,
                leftIndent=0.5 * cm,
                bulletIndent=0.5 * cm,
            )
        else:
            para_style = ParagraphStyle(
                'NormalText',
                parent=styles['Normal'],
                fontSize=14,
                fontName=_register_fonts.regular_font,  # Используем зарегистрированный TTF-шрифт
                leading=18,
                leftIndent=0.5 * cm,
                bulletIndent=0.5 * cm,
            )
        
        # Экранируем HTML-специальные символы для Paragraph
        line_escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        # Создаем Paragraph с маркером списка
        para = Paragraph(f"• {line_escaped}", para_style)
        para.wrapOn(c, width, max_height)
        
        # Проверяем, помещается ли параграф
        if current_y - para.height < 4 * cm:
            break
        
        para.drawOn(c, x, current_y - para.height)
        current_y -= para.height
        
        # Не вываливаемся за нижний край
        if current_y < 4 * cm:
            break


def _draw_table(c: canvas.Canvas, content: str, x: float, y: float, width: float, max_height: float):
    """Рисует таблицу на PDF"""
    if not content or not content.strip():
        return
    
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    if not lines:
        return
    
    # Парсим таблицу
    table_data = []
    for line in lines:
        # Убираем маркеры списка и markdown форматирование
        clean_line = line.strip()
        if clean_line.startswith('•') or clean_line.startswith('-') or clean_line.startswith('*'):
            clean_line = clean_line[1:].strip()
        clean_line = clean_line.replace('```', '').replace('`', '').replace('**', '').replace('*', '')
        
        # Проверяем наличие разделителя |
        if '|' in clean_line:
            cells = [cell.strip() for cell in clean_line.split('|')]
            # Фильтруем пустые ячейки
            cells = [cell for cell in cells if cell]
            if len(cells) >= 2:  # Минимум 2 столбца
                table_data.append(cells)
    
    if not table_data or len(table_data) < 2:
        # Если не таблица, ничего не добавляем
        logger.warning(f"Не удалось распарсить таблицу для PDF (найдено строк: {len(table_data)}), пропускаем")
        return
    
    # Определяем размеры ячеек
    num_cols = len(table_data[0])
    num_rows = len(table_data)
    cell_width = width / num_cols
    cell_height = min(max_height / num_rows, 1.5 * cm)
    
    current_y = y
    
    for row_idx, row in enumerate(table_data):
        current_x = x
        
        for col_idx in range(num_cols):
            cell_text = row[col_idx] if col_idx < len(row) else ""
            
            # Стили для заголовка (цвета и рамки)
            if row_idx == 0:
                c.setFillColor(colors.HexColor('#4472C4'))  # Синий
                c.rect(current_x, current_y - cell_height, cell_width, cell_height, fill=1, stroke=1)
                c.setFillColor(colors.white)
            else:
                c.setFillColor(colors.white)
                c.rect(current_x, current_y - cell_height, cell_width, cell_height, fill=0, stroke=1)
                c.setFillColor(colors.black)
            
            # Обрезаем текст если слишком длинный
            max_chars = int(cell_width / (0.3 * cm))
            if len(cell_text) > max_chars:
                cell_text = cell_text[:max_chars-3] + "..."
            
            # Используем Paragraph с TTF-шрифтом для поддержки Unicode/кириллицы
            styles = getSampleStyleSheet()
            if row_idx == 0:
                cell_style = ParagraphStyle(
                    'TableHeader',
                    parent=styles['Normal'],
                    fontSize=12,
                    fontName=_register_fonts.bold_font,  # Используем зарегистрированный TTF-шрифт
                    leading=14,
                    alignment=1,  # Центрирование
                )
            else:
                cell_style = ParagraphStyle(
                    'TableCell',
                    parent=styles['Normal'],
                    fontSize=11,
                    fontName=_register_fonts.regular_font,  # Используем зарегистрированный TTF-шрифт
                    leading=13,
                    alignment=1,  # Центрирование
                )
            
            # Экранируем HTML-специальные символы
            cell_text_escaped = cell_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            cell_para = Paragraph(cell_text_escaped, cell_style)
            cell_para.wrapOn(c, cell_width, cell_height)
            
            # Рисуем по центру ячейки
            text_x = current_x
            text_y = current_y - cell_height / 2 - cell_para.height / 2
            
            cell_para.drawOn(c, text_x, text_y)
            
            current_x += cell_width
        
        current_y -= cell_height
        
        if current_y < 4 * cm:
            break


def _draw_chart(c: canvas.Canvas, content: str, x: float, y: float, width: float, max_height: float):
    """Рисует данные графика на PDF (пока как структурированный список)"""
    if not content or not content.strip():
        return
    
    # Парсим данные графика
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    chart_data = {}
    
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
                # Если не число, используем как есть
                chart_data[name] = value_str
    
    if not chart_data:
        logger.warning("Не удалось распарсить данные графика для PDF, пропускаем")
        return
    
    # Форматируем как список
    formatted_lines = []
    for name, value in chart_data.items():
        formatted_lines.append(f"{name}: {value}")
    
    formatted_content = "\n".join(formatted_lines)
    _draw_text_content(c, formatted_content, x, y, width, max_height)


def _draw_image(c: canvas.Canvas, image_path: str, x: float, y: float, width: float, max_height: float):
    """Рисует изображение в той же области, что и текст (y — верхняя граница)"""
    if not image_path:
        return
    try:
        c.drawImage(
            image_path,
            x,
            y - max_height,
            width=width,
            height=max_height,
            preserveAspectRatio=True,
            anchor="nw",
        )
    except Exception as e:
        logger.warning(f"Не удалось добавить изображение '{image_path}': {e}")
