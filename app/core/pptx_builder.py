# app/core/pptx_builder.py
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import PP_PLACEHOLDER
import io
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class PresentationBuilder:
    def __init__(self, template_path: Optional[str] = None):
        if template_path:
            self.prs = Presentation(template_path)
            self.layout_map = {}
            for i, layout in enumerate(self.prs.slide_layouts):
                name = layout.name.lower()
                if 'title and content' in name:
                    self.layout_map['title_and_content'] = i
                elif 'two content' in name:
                    self.layout_map['two_content'] = i
                elif 'title only' in name:
                    self.layout_map['title_only'] = i
                elif 'title slide' in name:
                    self.layout_map['title'] = i
            self.clear_slides()
        else:
            self.prs = Presentation()

    def clear_slides(self):
        """Очищает все слайды из шаблона, но сохраняет дизайн/layout'ы"""
        try:
            sldIdLst = self.prs.slides._sldIdLst
            for sldId in list(sldIdLst):
                sldIdLst.remove(sldId)
            logger.info("✅ Шаблон очищен")
        except Exception as e:
            logger.warning(f"Не удалось очистить шаблон: {e}")
            self.prs = Presentation()

    def add_slide(self, slide_type: str, title: str, content: str,
                  images: Optional[List[str]] = None):
        """
        Add a new slide to the presentation, choosing a layout dynamically based on
        slide_type, the presence of images, and whether there is any content text.

        Parameters:
        - slide_type: "title" or "content". Title slides always use the title layout.
        - images: A list of image paths. The number of images determines if a "two content" or "title only" layout is chosen.
        - content: The text for the slide. If content is empty but an image is present, a "title only" layout is chosen.
        """
        images = images or []
        # Determine layout index dynamically
        if slide_type == "title":
            layout_idx = 0
        else:
            if len(images) == 0:
                layout_idx = 1  # title and content
            elif len(images) >= 2:
                layout_idx = 2  # two content
            else:
                # len(images) == 1
                if content and content.strip():
                    layout_idx = 2  # two content (one image and text)
                else:
                    layout_idx = 3  # title only (one image and no text)

        try:
            slide = self.prs.slides.add_slide(self.prs.slide_layouts[layout_idx])
        except Exception:
            slide = self.prs.slides.add_slide(self.prs.slide_layouts[0])

        # Заголовок
        if slide.shapes.title:
            slide.shapes.title.text = title

        # Контент — пытаемся найти BODY плейсхолдер
        body_shape = None
        try:
            for ph in slide.placeholders:
                if ph.placeholder_format.type == PP_PLACEHOLDER.BODY:
                    body_shape = ph
                    break
        except Exception:
            body_shape = None

        if body_shape is not None:
            tf = body_shape.text_frame
            tf.clear()
            p0 = tf.paragraphs[0]
            p0.text = content
            p0.font.size = Pt(18)
            p0.alignment = PP_ALIGN.LEFT
        else:
            # Фоллбек: текстбокс
            left = Inches(1)
            top = Inches(2)
            width = Inches(11)
            height = Inches(4)

            textbox = slide.shapes.add_textbox(left, top, width, height)
            tf = textbox.text_frame
            tf.text = content
            for p in tf.paragraphs:
                p.font.size = Pt(18)
                p.alignment = PP_ALIGN.LEFT

        # Картинки (до 4, сетка 2x2)
        if images:
            left0 = Inches(1)
            top0 = Inches(6.0)
            cell_w = Inches(5.5)
            cell_h = Inches(1.6)
            max_imgs = min(4, len(images))

            for i, img_path in enumerate(images[:max_imgs]):
                row, col = divmod(i, 2)
                left = left0 + col * (cell_w + Inches(0.3))
                top = top0 + row * (cell_h + Inches(0.3))
                try:
                    slide.shapes.add_picture(img_path, left, top, width=cell_w, height=cell_h)
                except Exception as e:
                    logger.warning(f"Не удалось добавить изображение '{img_path}': {e}")

        return slide

    def save_to_bytes(self) -> io.BytesIO:
        bytes_io = io.BytesIO()
        self.prs.save(bytes_io)
        bytes_io.seek(0)
        return bytes_io

    def get_slide_count(self) -> int:
        return len(self.prs.slides)



