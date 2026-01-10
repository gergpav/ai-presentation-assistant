from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    LLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
    MAX_NEW_TOKENS: int = 150
    TEMPERATURE: float = 0.3

    DATABASE_URL_ASYNC: str
    DATABASE_URL_SYNC: str

    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    class Config:
        env_file = ".env"



settings = Settings()