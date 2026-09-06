from dataclasses import dataclass, field

from dotenv import dotenv_values


@dataclass
class Dotenv:
    api_endpoint: str = field(default_factory=lambda: dotenv_values()["api_endpoint"])
    api_key: str = field(default_factory=lambda: dotenv_values()["api_key"])
