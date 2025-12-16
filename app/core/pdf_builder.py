# app/core/pdf_builder.py
import io
import logging
from typing import List

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.units import cm

from app.utils.helpers import SlideExport

logger = logging.getLogger(__name__)


def slides_to_pdf_bytes(slides: List[SlideExport], audience: str) -> bytes:
    """
    Генерация PDF-презентации напрямую из структуры слайдов.
    Каждый слайд = отдельная страница PDF.
    """
    buffer = io.BytesIO()

    # Горизонтальная страница формата A4 — условно "слайд"
    page_size = landscape(A4)
    c = canvas.Canvas(buffer, pagesize=page_size)
    width, height = page_size

    for idx, slide in enumerate(slides, start=1):
        # --- Заголовок ---
        title = slide.title or "Слайд"
        c.setFont("Helvetica-Bold", 24)
        c.drawString(2 * cm, height - 2 * cm, title)

        # --- Подзаголовок: аудитория + номер слайда ---
        subtitle = f"Аудитория: {audience} • Слайд {idx} из {len(slides)}"
        c.setFont("Helvetica", 10)
        c.drawString(2 * cm, height - 3 * cm, subtitle)

        # --- Основной текст (буллеты) ---
        text_y = height - 4 * cm
        c.setFont("Helvetica", 14)

        # content уже приходит как набор строк с \n,
        # упрощённо считаем, что пользователь не делает по 300 символов в строке
        for raw_line in (slide.content or "").split("\n"):
            line = raw_line.strip()
            if not line:
                continue

            # гарантируем буллет
            if line.startswith("•"):
                bullet_text = line
            else:
                bullet_text = f"• {line}"

            c.drawString(2.5 * cm, text_y, bullet_text)
            text_y -= 0.9 * cm

            # не вываливаемся за нижний край
            if text_y < 4 * cm:
                break

        # --- Картинки (если используешь images) ---
        # Предполагаем, что slide.images: List[str] — пути к файлам
        images = getattr(slide, "images", []) or []

        if images:
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
