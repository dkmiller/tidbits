from dataclasses import dataclass
from pathlib import Path

from dirhash import dirhash


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class ImageClient:
    """
    Detect Docker image definitions.
    """

    root: str
