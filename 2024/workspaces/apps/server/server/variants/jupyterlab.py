from server.variants.base import AbstractVariant


class Jupyterlab(AbstractVariant):
    def args(self, port: int):
        return [
            "jupyter",
            "lab",
            "--allow-root",
            "--ip=0.0.0.0",
            f"--port={port}",
            "--IdentityProvider.token=''",
        ]

    def name(self) -> str:
        return "jupyterlab"

    def readiness(self):
        return "/api"
