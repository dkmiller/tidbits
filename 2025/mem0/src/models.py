from dataclasses import dataclass


@dataclass
class Config:
    agent_id: str
    embedding_dims: int
    model_id: str
    system_prompt: str

    env: dict[str, str]

    # TODO: it's silly that these are all caps.
    BRAINTRUST_API_KEY: str
    OPENAI_API_KEY: str
