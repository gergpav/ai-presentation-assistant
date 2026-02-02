from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    LLM_MODEL: str = "Qwen/Qwen2.5-3B-Instruct"
    MAX_NEW_TOKENS: int = 200  
    TEMPERATURE: float = 0.3
    USE_QUANTIZATION: bool = False

    # Таймаут генерации контента (в секундах)
    LLM_GENERATION_TIMEOUT_SEC: int = 600

    # Количество параллельных воркеров для генерации слайдов
    WORKER_PARALLEL_JOBS: int = 3

    # Настройки локального Stable Diffusion для генерации изображений
    STABLE_DIFFUSION_MODEL_ID: str = "stabilityai/stable-diffusion-xl-base-1.0"
    # Устройство для генерации изображений: 'cuda' для GPU, 'cpu' для CPU, или None для автоопределения
    STABLE_DIFFUSION_DEVICE: Optional[str] = None
    # Количество шагов генерации (больше = качественнее, но медленнее)
    STABLE_DIFFUSION_STEPS: int = 30
    # Сила следования промпту (7.5 - стандартное значение)
    STABLE_DIFFUSION_GUIDANCE_SCALE: float = 7.5
    # Размер изображения (512x512 - стандарт для SD 1.5, 1024x1024 для SDXL)
    STABLE_DIFFUSION_WIDTH: int = 640
    STABLE_DIFFUSION_HEIGHT: int = 640

    DATABASE_URL_ASYNC: str
    DATABASE_URL_SYNC: str

    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    class Config:
        env_file = ".env"
        extra = "ignore"



settings = Settings()