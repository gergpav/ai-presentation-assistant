from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    LLM_MODEL: str = "Qwen/Qwen2.5-3B-Instruct"
    MAX_NEW_TOKENS: int = 250
    TEMPERATURE: float = 0.3
    # Использовать 8-bit квантование для моделей 7B+ на CPU
    # Уменьшает потребление памяти в 2 раза, скорость почти не страдает
    # Требует установки: pip install bitsandbytes
    # ВАЖНО: Квантование на CPU требует специальных настроек и может не работать
    # Для модели 3B рекомендуется отключить квантование
    USE_QUANTIZATION: bool = False

    # Таймаут генерации контента (в секундах)
    # По умолчанию 600 секунд (10 минут) для генерации одного слайда
    # Можно изменить через переменную окружения LLM_GENERATION_TIMEOUT_SEC
    LLM_GENERATION_TIMEOUT_SEC: int = 600

    # Количество параллельных воркеров для генерации слайдов
    # По умолчанию 3 - позволяет генерировать до 3 слайдов одновременно
    # Можно изменить через переменную окружения WORKER_PARALLEL_JOBS
    WORKER_PARALLEL_JOBS: int = 3

    # Настройки локального Stable Diffusion для генерации изображений
    # Модель для генерации (по умолчанию Stable Diffusion 1.5)
    # Популярные модели: 
    #   - runwayml/stable-diffusion-v1-5 (быстрая, ~4GB VRAM)
    #   - stabilityai/stable-diffusion-2-1 (лучше качество, ~5GB VRAM)
    #   - stabilityai/stable-diffusion-xl-base-1.0 (лучшее качество, ~7GB VRAM)
    STABLE_DIFFUSION_MODEL_ID: str = "runwayml/stable-diffusion-v1-5"
    # Устройство для генерации изображений: 'cuda' для GPU, 'cpu' для CPU, или None для автоопределения
    # ВАЖНО: Если FORCE_CPU=true, GPU будет недоступен даже при указании 'cuda'
    # Рекомендация: если у вас мало VRAM (<12GB), используйте GPU только для одной модели
    STABLE_DIFFUSION_DEVICE: Optional[str] = None
    # Количество шагов генерации (больше = качественнее, но медленнее)
    STABLE_DIFFUSION_STEPS: int = 30
    # Сила следования промпту (7.5 - стандартное значение)
    STABLE_DIFFUSION_GUIDANCE_SCALE: float = 7.5
    # Размер изображения (512x512 - стандарт для SD 1.5, 1024x1024 для SDXL)
    STABLE_DIFFUSION_WIDTH: int = 512
    STABLE_DIFFUSION_HEIGHT: int = 512

    DATABASE_URL_ASYNC: str
    DATABASE_URL_SYNC: str

    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    class Config:
        env_file = ".env"
        extra = "ignore"



settings = Settings()