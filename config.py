from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.deepseek.com"
    OPENAI_MODEL: str = "deepseek-chat"
    
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    EMBEDDING_DIMENSION: int = 1536
    
    DATA_PATH: str = "./data/comments.csv"
    VECTOR_STORE_PATH: str = "./vector_store"
    LOG_PATH: str = "./logs"
    
    TOP_K_RETRIEVAL: int = 10
    MAX_TOKENS: int = 2000

    @property
    def LLM_API_KEY(self) -> str:
        return self.DEEPSEEK_API_KEY or self.OPENAI_API_KEY

    @property
    def LLM_BASE_URL(self) -> str:
        return self.DEEPSEEK_BASE_URL or self.OPENAI_BASE_URL

    @property
    def LLM_MODEL(self) -> str:
        return self.DEEPSEEK_MODEL or self.OPENAI_MODEL
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
