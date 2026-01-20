# КРИТИЧНО: Устанавливаем переменные окружения ДО импорта PyTorch/transformers
# для предотвращения попыток использования CUDA
import os

# Принудительно отключаем CUDA (по умолчанию true для избежания ошибок)
force_cpu = os.getenv("FORCE_CPU", "true").lower() == "true"
if force_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Скрываем GPU от всех библиотек
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.api.auth import router as auth_router
from app.api.projects import router as projects_router
from app.api.slides import router as slides_router
from app.api.generate import router as generate_router
from app.api.documents import router as documents_router
from app.api.export import router as export_router
from app.api.jobs import router as jobs_router
from app.api.download import router as download_router
from app.api.templates import router as templates_router
from app.core.embeddings import document_index, model
from app.core.llm_generator import content_generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 AI Presentation Assistant starting up...")
    health = content_generator.health_check()
    logger.info(f"LLM Model status: {health}")
    yield
    # Shutdown
    logger.info("🛑 AI Presentation Assistant shutting down...")


app = FastAPI(
    title="AI Presentation Assistant",
    description="Сервис для автоматической генерации инвестиционных презентаций",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Нужно, чтобы фронтенд мог прочитать Content-Disposition и взять filename с расширением
    expose_headers=["Content-Disposition"],
)

# Auth
app.include_router(auth_router)

# Projects
app.include_router(projects_router)

# Slides
app.include_router(slides_router)
app.include_router(generate_router)
app.include_router(documents_router)

# Jobs
app.include_router(jobs_router)

# Export
app.include_router(export_router)
app.include_router(download_router)

# Templates
app.include_router(templates_router)


@app.get("/")
def root():
    return {
        "message": "AI Presentation Assistant is running 🚀",
    }


@app.get("/health")
def health_check():
    """Проверка здоровья всех компонентов системы"""
    model_health = content_generator.health_check()

    # Безопасная проверка document_index
    try:
        documents_loaded = len(document_index.documents) > 0
        documents_count = len(document_index.documents)
        index_built = document_index.is_built
    except Exception as e:
        logger.error(f"Error checking document index: {e}")
        documents_loaded = False
        documents_count = 0
        index_built = False

    logger.info(f"Model device: {model.device}")

    return {
        "status": "healthy" if model_health["status"] in ["healthy", "loaded"] else "degraded",
        "components": {
            "llm_model": model_health,
            "document_index": {
                "loaded": documents_loaded,
                "documents_count": documents_count,
                "index_built": index_built
            }
        }
    }