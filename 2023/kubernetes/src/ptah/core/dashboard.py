import base64
import webbrowser
from dataclasses import dataclass

import pyperclip
from injector import inject
from rich import print

from ptah.core.shell import ShellClient


@inject
@dataclass
class Dashboard:
    shell: ShellClient

    def spawn(self, namespace: str, user: str) -> None:
        # TODO: this is pretty broken.
        token = self.shell("kubectl", "-n", namespace, "create", "token", user)
        url = "http://localhost:8001/api/v1/namespaces/default/services/https:kubernetes-dashboard:https/proxy/"

        print(f"Copy/pasting the token below and opening the URL:\n\n\t{token}\n\n\t{url}\n")

        pyperclip.copy(token)
        webbrowser.open(url)

    def grafana(self, namespace: str):
        password_b64 = self.shell(
            "kubectl",
            "get",
            "secret",
            "--namespace",
            namespace,
            "grafana",
            "-o",
            "jsonpath={.data.admin-password}",
        )
        url = "http://localhost:3000"

        # https://www.askpython.com/python/examples/decoding-base64-data
        password_encoded = password_b64.encode("ascii")
        decoded_bytes = base64.b64decode(password_encoded)
        password = decoded_bytes.decode("ascii")

        print(f"Copy/pasting the password below and opening the URL:\n\n\t{password}\n\n\t{url}\n")

        pyperclip.copy(password)
        webbrowser.open(url)
