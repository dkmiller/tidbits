from dataclasses import dataclass


@dataclass
class Config:
    openai_api_key: str
    model_id: str
    system_prompt: str
