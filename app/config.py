from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    LLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
    MAX_NEW_TOKENS: int = 64
    TEMPERATURE: float = 0.3


settings = Settings()