from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging
from app.api import upload, generate, presentation_templates
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


# Подключаем только upload и generate
app.include_router(upload.router, prefix="/document", tags=["document"])
app.include_router(generate.router, prefix="/presentation", tags=["presentation"])
app.include_router(presentation_templates.router, prefix="/templates", tags=["presentation_templates"])


@app.get("/")
def root():
    return {
        "message": "AI Presentation Assistant is running 🚀",
        "endpoints": {
            "upload": "/upload",
            "generate": "/generate/presentation",
            "llm_test": "/generate/test-llm",
            "llm_status": "/generate/llm-status"
        }
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