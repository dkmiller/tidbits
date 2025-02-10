from abc import ABC, abstractmethod

from fastapi import HTTPException, status


class AbstractVariant(ABC):
    """
    Imitate:
    https://github.com/dkmiller/tidbits/blob/main/2023/kubernetes/examples/e2e/pods/api.yaml

    TODO: replace this with a Jinja-templated YAML format?
    """

    @abstractmethod
    def args(self, port: int) -> list[str]:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def readiness(self) -> str:
        pass

    @classmethod
    def variant(cls, image_alias: str):
        # https://stackoverflow.com/a/3862957
        all_instances = [sub() for sub in cls.__subclasses__()]
        instances = [i for i in all_instances if i.name() == image_alias]
        if instances:
            return instances[0]
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"Unsupported image alias {image_alias}"
        )
