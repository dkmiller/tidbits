from dataclasses import dataclass, field
from pathlib import Path

from cachelib import FileSystemCache
from injector import inject
from omegaconf import DictConfig, OmegaConf
from rich import print

from ptah.core.shell import ShellClient

# TODO: why aren't default values being DI'd properly?
PATH = "kind.yaml"


@inject
@dataclass
class Kind:
    cache: FileSystemCache
    shell: ShellClient

    @property
    def cluster(self) -> str:
        kind_config: DictConfig = OmegaConf.load(PATH)  # type: ignore
        return kind_config.get("name", "kind")

    def create(self):
        path = Path(PATH)
        if not path.is_file():
            return

        cluster_name = self.cluster

        if self.cache.has(f"kind__{cluster_name}"):
            return

        clusters = self.shell("kind", "get", "clusters")

        if cluster_name in clusters:
            print(f"Kind cluster '{cluster_name}' is already up")
        else:
            print(f"Creating Kind cluster '{cluster_name}'")
            self.shell("kind", "create", "cluster", f"--config={path}")

        self.cache.set(f"kind__{cluster_name}", "any")
