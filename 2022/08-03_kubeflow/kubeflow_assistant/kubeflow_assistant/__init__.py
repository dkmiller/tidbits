"""
Utilities for quickly iterating on local Kubeflow pipeline definitions.
"""

import docker
from docker.errors import BuildError
import functools
import kfp.components as comp
import logging
from pathlib import Path
import yaml


log = logging.getLogger(__name__)


class KubeflowAssistant:
    def __init__(self, root: Path):
        self.root = root
        self.docker_client = docker.from_env()

    def component_path(self, name: str) -> Path:
        """
        Override this if different convention.
        """
        rv = self.root / f"components/{name}/component.yaml"
        return rv

    def component_docker_path(self, name: str) -> str:
        """
        Path to local Docker context for a component. Override this to change
        the convention.
        """
        return str(self.component_path(name).parent.absolute())

    @functools.lru_cache(maxsize=None)
    def build_and_load_component(self, name: str):
        path = self.component_path(name)
        text = path.read_text()
        parsed = yaml.load(text, Loader=yaml.SafeLoader)
        image_tag = parsed["implementation"]["container"]["image"]

        build_path = self.component_docker_path(name)

        log.info(f"Building Docker image {image_tag} from path {build_path}")

        try:
            image, _ = self.docker_client.images.build(path=build_path, tag=image_tag)  # type: ignore
        except BuildError as e:
            debug_command = f"docker build -t {image_tag} {build_path}"
            raise RuntimeError(
                f"Docker build failed! See failure message above, debug with the command\n\n\t{debug_command}"
            ) from e

        try:
            digest = image.attrs["RepoDigests"][0]  # type: ignore
        except IndexError:
            digest = image_tag
            log.warning(f"Pushing {image_tag} for the first time")

        log.info(f"Pushing {image_tag}")
        r = self.docker_client.images.push(image_tag)
        log.info(f"Push result: {r}")

        # TODO: don't always do this.
        # TODO: do this in the parsed YAML
        text = text.replace(image_tag, digest)
        log.debug(f"Raw component text\n\n{text}\n\n")

        rv = comp.load_component_from_text(text)
        return rv
