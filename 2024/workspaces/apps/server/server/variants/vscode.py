from server.variants.base import AbstractVariant


class VsCode(AbstractVariant):
    def args(self, port: int):
        return [
            "code-server",
            "--bind-addr",
            f"0.0.0.0:{port}",
            "--auth",
            "none",
        ]

    def name(self) -> str:
        return "vscode"

    def readiness(self):
        return "/healthz"
