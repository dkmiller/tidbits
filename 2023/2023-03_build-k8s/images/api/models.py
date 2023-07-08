from dataclasses import dataclass


@dataclass
class CatFact:
    fact: str
    length: int


@dataclass
class Health:
    ok: bool
