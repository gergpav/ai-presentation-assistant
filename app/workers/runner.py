# app/workers/runner.py
import asyncio
import logging
import uuid
import os
from sqlalchemy.orm import Session, joinedload
from pathlib import Path
from typing import Optional

import pdfplumber
from docx import Document as DocxDocument
from pptx import Presentation as PptxPresentation
import pandas as pd

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.db.models.job import Job
from app.db.models.project import Project
from app.db.models.slide import Slide
from app.db.models.slide_content import SlideContent
from app.db.models.slide_document import SlideDocument
from app.db.models.file import File
from app.db.models.enums import JobStatus, JobType, SlideStatus, FileKind
from app.db.models import Template

from app.core.llm_generator import content_generator
from app.core.embeddings import DocumentIndex

from app.core.pptx_builder import PresentationBuilder
from app.core.pdf_builder import slides_to_pdf_bytes
from app.utils.helpers import SlideExport

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

POLL_INTERVAL_SEC = 1.0
STORAGE_DIR = Path("storage")
STORAGE_DIR.mkdir(exist_ok=True)


# ----------------------------
# Helpers: SlideContent text field (под разные имена колонок)
# ----------------------------
def _sc_get_text(sc: SlideContent | None) -> str:
    if sc is None:
        return ""
    if hasattr(sc, "content_text"):
        return sc.content_text or ""
    if hasattr(sc, "content"):
        val = sc.content
        # В нашей схеме SlideContent.content = JSON (dict), чаще всего {"text": "..."}.
        # Экспорт ожидает строку.
        if isinstance(val, dict):
            if "text" in val and isinstance(val.get("text"), str):
                return val["text"]
            # fallback: сериализуем в строку, чтобы не падать в Pydantic
            try:
                import json
                return json.dumps(val, ensure_ascii=False)
            except Exception:
                return str(val)
        return val or ""
    if hasattr(sc, "text"):
        return sc.text or ""
    return ""


def _sc_set_text(sc: SlideContent, text: str) -> None:
    if hasattr(sc, "content_text"):
        sc.content_text = text
        return
    if hasattr(sc, "content"):
        sc.content = text
        return
    if hasattr(sc, "text"):
        sc.text = text
        return
    raise RuntimeError("SlideContent: не найдено поле для текста (content_text/content/text)")


# ----------------------------
# Helpers: parsing files from path (pptx/docx/xlsx/pdf)
# ----------------------------
async def _parse_text_from_path(path: str) -> str:
    """
    Мини-парсер "как есть" для контекста.
    Хранит итоговый текст в SlideDocument.parsed_text, чтобы не парсить повторно.
    """
    p = Path(path)
    if not p.exists():
        return ""

    ext = p.suffix.lower()

    try:
        if ext == ".pdf":
            text = ""
            with pdfplumber.open(str(p)) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
            return text.strip()

        if ext == ".docx":
            doc = DocxDocument(str(p))
            parts = [par.text for par in doc.paragraphs if par.text]
            return "\n".join(parts).strip()

        if ext == ".pptx":
            pres = PptxPresentation(str(p))
            parts: list[str] = []
            for slide in pres.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        parts.append(shape.text)
            return "\n".join(parts).strip()

        if ext in (".xlsx", ".xls"):
            # читаем все листы, склеиваем как текстовую таблицу
            xls = pd.ExcelFile(str(p))
            out: list[str] = []
            for sheet in xls.sheet_names:
                df = xls.parse(sheet)
                out.append(f"=== {sheet} ===")
                out.append(df.to_string(index=False))
            return "\n".join(out).strip()

        # fallback: просто читаем как текст (если вдруг txt)
        return p.read_text(encoding="utf-8", errors="ignore").strip()

    except Exception as e:
        logger.warning(f"Failed to parse {path}: {e}")
        return ""


# ----------------------------
# Helpers: build context for a slide from SlideDocument
# ----------------------------
async def _build_slide_context(db: AsyncSession, slide_id: int, query: str) -> str:
    res = await db.execute(
        select(SlideDocument)
        .where(SlideDocument.slide_id == slide_id)
        .order_by(SlideDocument.id.asc())
    )
    docs = res.scalars().all()
    if not docs:
        return ""

    documents_for_index: list[dict] = []

    for d in docs:
        if d.parsed_text:
            text = d.parsed_text
        else:
            text = await _parse_text_from_path(d.storage_path)
            if text:
                d.parsed_text = text  # кешируем
        documents_for_index.append({"text": text, "metadata": {"filename": d.filename}})

    await db.commit()

    idx = DocumentIndex()
    idx.add_documents(documents_for_index)
    idx.build_index()

    hits = idx.search(query, k=5)
    context = "\n\n".join(f"[{source}]\n{content}" for (content, _kind, source) in hits if content)
    return context[:12000]


async def _get_latest_slide_content(db: AsyncSession, slide_id: int) -> SlideContent | None:
    res = await db.execute(
        select(SlideContent)
        .where(SlideContent.slide_id == slide_id)
        .order_by(desc(SlideContent.id))
        .limit(1)
    )
    return res.scalar_one_or_none()


# ----------------------------
# Jobs: export template (optional)
# ----------------------------
async def _resolve_project_template_path(db: AsyncSession, project_id: int) -> str | None:
    """
    Возвращает путь к pptx-шаблону проекта (если назначен).
    Поддерживает оба варианта:
      1) relationship: Project.template
      2) FK: Project.template_id
    """
    # 1) пробуем загрузить проект сразу с template (если relationship есть)
    try:
        res = await db.execute(
            select(Project)
            .options(joinedload(Project.template))
            .where(Project.id == project_id)
        )
        project = res.scalar_one()
        tmpl = getattr(project, "template", None)
        if tmpl and getattr(tmpl, "storage_path", None):
            return tmpl.storage_path
    except Exception:
        # relationship может отсутствовать — ок, падаем на fallback
        pass

    # 2) fallback: template_id -> Template
    res = await db.execute(select(Project).where(Project.id == project_id))
    project = res.scalar_one()

    template_id = getattr(project, "template_id", None)
    if not template_id:
        return None

    res = await db.execute(select(Template).where(Template.id == template_id))
    tmpl = res.scalar_one_or_none()
    if not tmpl:
        return None

    return getattr(tmpl, "storage_path", None)



# ----------------------------
# Job handlers
# ----------------------------
async def generate_slide_job(db: AsyncSession, job: Job) -> None:
    if job.slide_id is None:
        raise ValueError("Job.slide_id is required for generate_slide")

    slide = (await db.execute(select(Slide).where(Slide.id == job.slide_id))).scalar_one()
    project = (await db.execute(select(Project).where(Project.id == slide.project_id))).scalar_one()

    if not slide.prompt or not slide.prompt.strip():
        raise ValueError("Slide prompt is empty")

    job.progress = 10
    await db.commit()

    context_text = await _build_slide_context(db, slide.id, slide.prompt)

    job.progress = 30
    await db.commit()

    # Генерация может быть очень долгой на CPU/GPU.
    # Чтобы job не "висел" бесконечно в UI, запускаем генерацию в отдельном потоке
    # и ограничиваем её по времени.
    job.progress = 40
    await db.commit()

    async def _run_generation():
        return content_generator.generate_from_prompt(
            user_prompt=slide.prompt,
            context=context_text,
            audience=str(project.audience_type),
        )

    # таймаут (сек) — можно переопределить через env LLM_GENERATION_TIMEOUT_SEC
    try:
        timeout_sec = int(os.getenv("LLM_GENERATION_TIMEOUT_SEC", "180"))
    except Exception:
        timeout_sec = 180

    out = await asyncio.wait_for(asyncio.to_thread(lambda: content_generator.generate_from_prompt(
        user_prompt=slide.prompt,
        context=context_text,
        audience=str(project.audience_type),
    )), timeout=timeout_sec)
    generated_text = (out.get("content") or "").strip()

    job.progress = 70
    await db.commit()

    sc = SlideContent(slide_id=slide.id)
    _sc_set_text(sc, generated_text)

    # если у тебя есть version — увеличим
    if hasattr(sc, "version"):
        res = await db.execute(
            select(SlideContent)
            .where(SlideContent.slide_id == slide.id)
            .order_by(desc(SlideContent.version))
            .limit(1)
        )
        last = res.scalar_one_or_none()
        sc.version = (last.version + 1) if last else 1

    db.add(sc)

    # обновим статус слайда (если есть)
    if hasattr(slide, "status"):
        slide.status = SlideStatus.ready

    await db.commit()


async def export_project_pptx_job(db, job: Job):
    project = (await db.execute(select(Project).where(Project.id == job.project_id))).scalar_one()

    slides = (await db.execute(
        select(Slide)
        .where(Slide.project_id == project.id)
        .order_by(Slide.position.asc())
    )).scalars().all()

    # шаблон опционален
    template_path = await _resolve_project_template_path(db, project.id)

    # если шаблон назначен, но файла нет — лучше явно упасть (чтобы не молча делать "без шаблона")
    if template_path and not Path(template_path).exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    builder = PresentationBuilder(template_path=template_path)

    export_slides: list[SlideExport] = []
    for i, s in enumerate(slides, start=1):
        sc = await _get_latest_slide_content(db, s.id)
        export_slides.append(SlideExport(title=s.title or f"Slide {i}", content=_sc_get_text(sc), images=[]))

    for se in export_slides:
        builder.add_slide(slide_type="content", title=se.title, content=se.content, images=se.images)

    data = builder.save_to_bytes().getvalue()

    safe_title = (project.title or "presentation").strip().replace(" ", "_")
    filename = f"{safe_title}_{uuid.uuid4().hex}.pptx"
    out_path = STORAGE_DIR / filename
    out_path.write_bytes(data)

    out_file = File(
        user_id=job.user_id,
        project_id=project.id,
        kind=FileKind.export_pptx,
        filename=filename,
        storage_path=str(out_path),
        size_bytes=len(data),
    )
    db.add(out_file)
    await db.commit()
    await db.refresh(out_file)

    job.result_file_id = out_file.id
    await db.commit()

async def export_project_pdf_job(db, job: Job):
    project = (await db.execute(select(Project).where(Project.id == job.project_id))).scalar_one()

    slides = (await db.execute(
        select(Slide)
        .where(Slide.project_id == project.id)
        .order_by(Slide.position.asc())
    )).scalars().all()

    export_slides: list[SlideExport] = []
    for i, s in enumerate(slides, start=1):
        sc = await _get_latest_slide_content(db, s.id)
        export_slides.append(SlideExport(title=s.title or f"Slide {i}", content=_sc_get_text(sc), images=[]))

    pdf_bytes = slides_to_pdf_bytes(export_slides, audience=str(project.audience_type))

    safe_title = (project.title or "presentation").strip().replace(" ", "_")
    filename = f"{safe_title}_{uuid.uuid4().hex}.pdf"
    out_path = STORAGE_DIR / filename
    out_path.write_bytes(pdf_bytes)

    out_file = File(
        user_id=job.user_id,
        project_id=project.id,
        kind=FileKind.export_pdf,
        filename=filename,
        storage_path=str(out_path),
        size_bytes=len(pdf_bytes),
    )
    db.add(out_file)
    await db.commit()
    await db.refresh(out_file)

    job.result_file_id = out_file.id
    await db.commit()


async def handle_job(db: AsyncSession, job: Job) -> None:
    if job.type == JobType.generate_slide:
        await generate_slide_job(db, job)
        return

    if job.type == JobType.export_pptx:
        await export_project_pptx_job(db, job)
        return

    if job.type == JobType.export_pdf:
        await export_project_pdf_job(db, job)
        return

    raise NotImplementedError(f"Job type not supported yet: {job.type}")


# ----------------------------
# Job state helpers
# ----------------------------
async def _fetch_one_queued_job(db: AsyncSession) -> Optional[Job]:
    q = (
        select(Job)
        .where(Job.status == JobStatus.queued)
        .order_by(Job.id.asc())
        .with_for_update(skip_locked=True)
        .limit(1)
    )
    res = await db.execute(q)
    return res.scalar_one_or_none()


async def _set_job_running(db: AsyncSession, job: Job) -> None:
    job.status = JobStatus.running
    job.progress = 1
    job.error_message = None
    await db.commit()


async def _set_job_done(db: AsyncSession, job: Job) -> None:
    job.status = JobStatus.done
    job.progress = 100
    await db.commit()


async def _set_job_failed(db: AsyncSession, job: Job, exc: Exception) -> None:
    job.status = JobStatus.error
    job.progress = 100
    job.error_message = str(exc)
    # если это job генерации — отметим и слайд как error, чтобы UI перестал крутиться
    try:
        if job.slide_id is not None:
            slide = (await db.execute(select(Slide).where(Slide.id == job.slide_id))).scalar_one_or_none()
            if slide is not None and hasattr(slide, "status"):
                slide.status = SlideStatus.error
    except Exception:
        # не мешаем падению job
        pass
    await db.commit()


# ----------------------------
# Worker loop
# ----------------------------
async def worker_loop() -> None:
    logger.info("🧵 Worker started")

    while True:
        async with AsyncSessionLocal() as db:
            job = await _fetch_one_queued_job(db)
            if not job:
                await asyncio.sleep(POLL_INTERVAL_SEC)
                continue

            logger.info(f"➡️ Picked job id={job.id} type={job.type}")

            try:
                await _set_job_running(db, job)
                await handle_job(db, job)
                await _set_job_done(db, job)
                logger.info(f"✅ Job done id={job.id}")
            except Exception as e:
                logger.exception(f"❌ Job failed id={job.id}: {e}")
                await _set_job_failed(db, job, e)


def main() -> None:
    asyncio.run(worker_loop())


if __name__ == "__main__":
    main()

