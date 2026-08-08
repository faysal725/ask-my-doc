import os

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
    env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"),
    env_file_encoding="utf-8",
)

    # Groq (LLM)
    groq_api_key: str = ""

    # Qdrant (vector DB)
    qdrant_url: str = ""
    qdrant_api_key: str = ""

    # Gemini (embeddings)
    gemini_api_key: str = ""

    # App
    app_env: str = "development"
    cors_origins: str = "http://localhost:3000"


settings = Settings()