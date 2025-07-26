from dataclasses import dataclass


@dataclass
class Config:
    # Embedding model can't handle more than 512 tokens.
    char_limit: int = 512
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
