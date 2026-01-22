from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Рекомендуется использовать Qwen2.5-3B-Instruct для экономии памяти
    # Для лучшего качества можно использовать:
    # - Qwen/Qwen2.5-7B-Instruct (требует больше памяти, ~14GB без квантования)
    # - meta-llama/Llama-3.2-3B-Instruct (альтернатива 3B, немного умнее)
    # Можно изменить через переменную окружения LLM_MODEL
    LLM_MODEL: str = "Qwen/Qwen2.5-3B-Instruct"
    # Увеличено до 250 для генерации более полного контента слайдов
    # Можно изменить через переменную окружения MAX_NEW_TOKENS (рекомендуется 200-300 для качественного контента)
    MAX_NEW_TOKENS: int = 250
    TEMPERATURE: float = 0.3
    # Использовать 8-bit квантование для моделей 7B+ на CPU
    # Уменьшает потребление памяти в 2 раза, скорость почти не страдает
    # Требует установки: pip install bitsandbytes
    # ВАЖНО: Квантование на CPU требует специальных настроек и может не работать
    # Для модели 3B рекомендуется отключить квантование
    USE_QUANTIZATION: bool = False

    DATABASE_URL_ASYNC: str
    DATABASE_URL_SYNC: str

    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    class Config:
        env_file = ".env"
        extra = "ignore"



settings = Settings()