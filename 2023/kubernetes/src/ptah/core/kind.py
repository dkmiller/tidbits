from dataclasses import dataclass
from pathlib import Path

from injector import inject

from ptah.core.shell import ShellClient


@inject
@dataclass(frozen=True)
class Kind:
    shell: ShellClient

    def create(self, config: str):
        path = Path(config)

        # TODO: detect cluster name, caching:
        # https://python.land/data-processing/python-yaml

        if path.is_file():
            try:
                self.shell("kind", "create", "cluster", f"--config={path.absolute()}")
            except BaseException:
                pass
