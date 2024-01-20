import os
from dataclasses import dataclass, field


@dataclass
class Config:
    token: str = field(default_factory=lambda: os.environ["SPACEBOTS_TOKEN"])
    colorscale: str = "Blugrn"
