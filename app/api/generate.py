import time
from typing import Optional, List

import io
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.core.llm_generator import content_generator
from app.api.presentation_templates import templates_store
from app.core.pptx_builder import PresentationBuilder

from app.utils.helpers import ExportRequest, SlideExport, generate_one_slide
from app.core.pdf_builder import slides_to_pdf_bytes


logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/export")
async def export_presentation(request: ExportRequest):
    """
    Экспорт:
    - генерим content для каждого слайда по prompt (+контекст)
    - собираем PPTX или PDF
    """
    if not request.slides:
        raise HTTPException(status_code=400, detail="Нет слайдов для экспорта")

    if not content_generator.is_loaded:
        raise HTTPException(status_code=503, detail="LLM-модель не загружена")

    t0 = time.time()
    logger.info(
        f"🚀 Export start: slides={len(request.slides)}, format={request.format}, audience={request.audience}"
    )

    # 1) Генерация всех слайдов
    generated: List[SlideExport] = []
    for i, slide in enumerate(request.slides):
        generated.append(await generate_one_slide(slide, request.audience))
        logger.info(f"    🟢 Slide {i+1}/{len(request.slides)} generated")

    t1 = time.time()
    logger.info(f"✅ Slides generated in {t1 - t0:.2f}s")

    # 2) PPTX
    if request.format == "pptx":
        template_path: Optional[str] = None
        if request.template_id:
            tmpl = templates_store.get(request.template_id)
            if not tmpl:
                raise HTTPException(status_code=404, detail="Шаблон не найден")
            template_path = tmpl["file_path"]

        builder = PresentationBuilder(template_path=template_path)

        for s in generated:
            builder.add_slide(
                slide_type="content",
                title=s.title,
                content=s.content,
                images=s.images,
            )

        pptx_io = builder.save_to_bytes()
        logger.info(f"📊 PPTX built in {time.time() - t1:.2f}s (total {time.time() - t0:.2f}s)")

        return StreamingResponse(
            pptx_io,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            headers={"Content-Disposition": 'attachment; filename="presentation.pptx"'},
        )

    # 3) PDF
    if request.format == "pdf":
        try:
            pdf_bytes = slides_to_pdf_bytes(generated, request.audience)
        except Exception as e:
            logger.error(f"Ошибка генерации PDF: {e}")
            raise HTTPException(status_code=500, detail="Ошибка при генерации PDF")

        logger.info(f"📄 PDF built in {time.time() - t1:.2f}s (total {time.time() - t0:.2f}s)")

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="presentation.pdf"'},
        )

    raise HTTPException(status_code=400, detail="Неподдерживаемый формат экспорта")
