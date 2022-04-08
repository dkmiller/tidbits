from dataclasses import dataclass


@dataclass
class Config:
    BLANK_WORD: str = "<blank>"
    BOS_WORD: str = "<s>"
    EOS_WORD: str = "</s>"
    english: str = "en_core_web_sm"
    german: str = "de_core_news_sm"

    MAX_LEN: int = 100

    model_dimension: int = 512
    d_ff: int = 2048
    h: int = 8
    dropout: float = 0.1
    N: int = 6
