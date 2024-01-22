"""
Load from JSON or dict with https://dataclass-wizard.readthedocs.io/en/latest/.

https://stackoverflow.com/a/72101312
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar

from dataclass_wizard import JSONWizard

_DEFAULT_TOKEN = os.environ.get("SPACEBOTS_TOKEN", "")


@dataclass
class Config:
    token: str = _DEFAULT_TOKEN
    colorscale: str = "Blugrn"


@dataclass(frozen=True)
class User:
    id: str
    name: str
    credits: int
    created_at: str
    registered: bool

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class Fleet(JSONWizard):
    id: str
    location_system_id: str
    ships: dict[str, int]
    cargo: dict[str, int]


@dataclass
class System(JSONWizard):
    id: str
    name: str
    x: int
    y: int
    neighboring_systems: list[str]
    asteroid: dict[str, str] = field(default_factory=dict)
    station: dict[str, str] = field(default_factory=dict)

    def __str__(self):
        rv = self.name
        if asteroid := self.asteroid:
            rv += " (🪨 " + " ".join(asteroid.values()) + " )"
        if self.station.get("directSell"):
            rv += " 💰"
        return rv


T = TypeVar("T")


class Action(ABC):
    # Abstract properties: https://stackoverflow.com/a/48710068
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def act(self, option: T):
        pass

    @abstractmethod
    def submenu(self) -> T:
        pass
