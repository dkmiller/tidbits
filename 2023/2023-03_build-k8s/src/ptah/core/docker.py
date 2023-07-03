from dataclasses import dataclass
from typing import List

from cachelib import FileSystemCache
from injector import inject
from rich import print

from ptah.core.image import ImageDefinition
from ptah.core.shell import ShellClient


@inject
@dataclass(frozen=True)
class DockerClient:
    cache: FileSystemCache
    image_definitions: List[ImageDefinition]
    shell: ShellClient

    def build(self) -> None:
        build = []
        skip = 0

        for image in self.image_definitions:
            if self.cache.has(f"build__{image.uri}"):
                skip += 1
            else:
                build.append(image)

        msg = f"Building {len(build)} Docker images"
        if skip:
            msg += f" ({skip} already built)"
        print(msg)

        for image in build:
            path = str(image.location.parent)
            self.shell.run(["docker", "build", "-t", image.uri, path])
            self.cache.set(f"build__{image.uri}", "any")

    def push(self) -> None:
        push = []
        skip = 0
        for image in self.image_definitions:
            if self.cache.has(f"push__{image.uri}"):
                skip += 1
            else:
                push.append(image)

        uris = [i.uri for i in push]

        msg = f"Pushing {len(uris)} images"
        if skip:
            msg += f" ({skip} already pushed)"
        print(msg)

        if push:
            # TODO: handle pushing to a remote registry.
            # https://codeberg.org/hjacobs/pytest-kind/src/branch/main/pytest_kind/cluster.py
            self.shell.run(["kind", "load", "docker-image"] + uris)
            for uri in uris:
                self.cache.set(f"push__{uri}", "any")
