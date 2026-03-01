from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    EMBEDDING_DIMENSION: int = 1536
    
    DATA_PATH: str = "./data/comments.csv"
    VECTOR_STORE_PATH: str = "./vector_store"
    LOG_PATH: str = "./logs"
    
    TOP_K_RETRIEVAL: int = 10
    MAX_TOKENS: int = 2000
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
