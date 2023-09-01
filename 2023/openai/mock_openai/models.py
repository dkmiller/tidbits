from dataclasses import dataclass
from enum import Enum


@dataclass
class Config:
    base: str
    api_key: str


class Mode(Enum):
    httpx = "httpx"
    requests = "requests"
    aiohttp_naive = "aiohttp_naive"
    aiohttp_nowith = "aiohttp_nowith"
    aiohttp_singleton = "aiohttp_singleton"
