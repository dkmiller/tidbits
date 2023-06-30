from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Iterable

from dirhash import dirhash
from injector import Module, provider


@dataclass
class ImageDefinition:
    """
    Local definition of Docker image.
    """

    location: Path
    algorithm: str = "md5"

    @property
    def name(self):
        # https://stackoverflow.com/a/35188296
        if self.location.stem.lower() == "dockerfile" and not self.location.suffix:
            return self.location.parent.name
        else:
            return self.location.stem

    @property
    def tag(self):
        return dirhash(str(self.location.parent.absolute()), self.algorithm)[:7]

    @property
    def uri(self):
        return f"{self.name}:{self.tag}"


@dataclass
class ImageClient(Module):
    """
    Detect Docker image definitions.
    """

    root: str

    @provider
    def image_definitions(self) -> Iterable[ImageDefinition]:
        with_stem = Path(self.root).rglob("*.[dD]ockerfile")
        without_stem = Path(self.root).rglob("[dD]ockerfile")
        return map(ImageDefinition, chain(with_stem, without_stem))
