import logging
import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    pinecone_api_key: str = Field(..., description="Pinecone API Key")
    pinecone_index_name: str = Field(default="repo-architect", description="Pinecone Serverless Index")
    pinecone_cloud: str = Field(default="aws")
    pinecone_region: str = Field(default="us-east-1")

    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    embedding_dimension: int = Field(default=384)
    llm_model_id: str = Field(default="Qwen/Qwen2.5-Coder-3B-Instruct")

    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    chunk_max_chars: int = Field(default=6000)

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def torch_dtype(self) -> torch.dtype:
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


settings = Settings()
