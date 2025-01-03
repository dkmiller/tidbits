from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import List

from dirhash import dirhash
from injector import Module, multiprovider


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
        dockerignore = self.location.parent / ".dockerignore"

        if dockerignore.exists():
            ignore = dockerignore.read_text().splitlines()
        else:
            ignore = None

        return dirhash(str(self.location.parent.absolute()), self.algorithm, ignore=ignore)[:7]

    @property
    def uri(self):
        return f"{self.name}:{self.tag}"


@dataclass
class ImageClient(Module):
    """
    Detect Docker image definitions.
    """

    root: str

    @multiprovider
    def image_definitions(self) -> List[ImageDefinition]:
        with_stem = Path(self.root).rglob("*.[dD]ockerfile")
        without_stem = Path(self.root).rglob("[dD]ockerfile")
        return list(map(ImageDefinition, chain(with_stem, without_stem)))
