from dataclasses import dataclass, field

from dotenv import dotenv_values
from pydantic import BaseModel


@dataclass
class Dotenv:
    api_endpoint: str = field(default_factory=lambda: dotenv_values()["api_endpoint"])
    api_key: str = field(default_factory=lambda: dotenv_values()["api_key"])
    model: str = field(default_factory=lambda: dotenv_values()["model"])


class StorySetup(BaseModel):
    title: str
    setting: str


@dataclass
class Story:
    setting: str
    chapters: list[str]
