from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import List
from urllib.parse import urlparse

from dirhash import dirhash
from dockerfile_parse import DockerfileParser
from injector import Module, multiprovider


@dataclass
class ImageDefinition:
    """
    Local definition of Docker image.
    """

    location: Path
    algorithm: str = "md5"

    @property
    def parser(self) -> DockerfileParser:
        return DockerfileParser(path=str(self.location))

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
        suffix = f"{self.name}:{self.tag}"
        source = self.parser.labels.get("org.opencontainers.image.source")
        if source and "github.com" in source:
            parsed = urlparse(source)
            return f"ghcr.io{parsed.path}/{suffix}"
        else:
            return suffix

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
