import asyncio
import logging
from typing import List, Optional, Literal
from pydantic import BaseModel
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool

from app.core.embeddings import document_index
from app.core.llm_generator import content_generator
from app.config import settings

logger = logging.getLogger(__name__)

LLM_SEMAPHORE = asyncio.Semaphore(1)  # чтобы не было параллельных генераций
LLM_TIMEOUT_SEC = settings.LLM_GENERATION_TIMEOUT_SEC  # Таймаут из конфигурации


# ============ МОДЕЛИ ============


class SlideInput(BaseModel):
    """
    Слайд из редактора (то, что хранит фронт и отправляет на /preview или /export):
    - title: заголовок
    - prompt: обязательный промпт для генерации текста слайда
    - images/docs: опциональные вложения (пути или URL/ID, как у тебя заведено)
    """
    title: str
    prompt: str
    images: List[str] = []
    docs: List[str] = []


class SlideExport(BaseModel):
    """
    Слайд после генерации (то, что возвращает /preview, и то, что идёт в PPTX/PDF билдеры):
    - title: заголовок
    - content: сгенерированный текст (буллеты)
    - images: картинки (если есть)
    - layout: тип макета слайда (title, two_content, title_only, title_and_content)
    - visual_type: тип визуализации (text, chart, table, image)
    """
    title: str
    content: str
    images: List[str] = []
    layout: str = "title_and_content"
    visual_type: str = "text"


class ExportRequest(BaseModel):
    """
    Запрос на генерацию презентации.
    slides[] содержит title+prompt, а контент генерируется на бэке.
    """
    audience: str  # "Инвесторы" / "Топ-менеджеры" / "Эксперты"
    format: Literal["pptx", "pdf"]
    template_id: Optional[str] = None
    slides: List[SlideInput]


async def build_context(slide: SlideInput) -> str:
    """
    Контекст из документов: если документов нет — вернём пустую строку.
    Сейчас ищем по всему индексу по prompt (или title).
    При желании можно потом фильтровать по slide.docs.
    """
    if not document_index.is_built or not document_index.documents:
        return ""

    query = (slide.prompt or slide.title).strip()
    if not query:
        return ""

    try:
        # search синхронный → выносим в threadpool, чтобы не блокировать FastAPI
        results = await run_in_threadpool(document_index.search, query, 3)
    except Exception as e:
        logger.error(f"Ошибка поиска контекста: {e}")
        return ""

    parts = [r[0] for r in results if r and r[0]]
    return "\n".join(parts)



async def generate_one_slide(slide: SlideInput, audience: str) -> SlideExport:
    """
    Генерация текста одного слайда: title+prompt (+контекст из документов) → content
    """
    if not slide.title.strip() or not slide.prompt.strip():
        # На бэке тоже защищаемся, хотя фронт и так не даст “Добавить слайд”
        raise HTTPException(status_code=400, detail="У каждого слайда должны быть title и prompt")

    context = await build_context(slide)

    try:
        async with LLM_SEMAPHORE:
            res = await asyncio.wait_for(
                asyncio.to_thread(
                    content_generator.generate_from_prompt,
                    slide.prompt,
                    context,
                    audience,
                ),
                timeout=LLM_TIMEOUT_SEC,
            )

        content = (res.get("content") or "").strip()
        if not content:
            content = "• (пустой ответ модели)"
    except asyncio.TimeoutError:
        logger.error(f"⏱️ Таймаут генерации для '{slide.title}'")
        content = "• Таймаут генерации. Уменьши MAX_NEW_TOKENS или используй более лёгкую модель."
    except Exception as e:
        logger.error(f"Ошибка LLM для слайда '{slide.title}': {e}")
        content = "• Не удалось сгенерировать содержимое слайда"

    return SlideExport(title=slide.title, content=content, images=slide.images or [])