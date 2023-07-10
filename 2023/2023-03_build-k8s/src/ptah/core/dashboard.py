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
        token = self.shell("kubectl", "-n", namespace, "create", "token", user)
        url = f"http://localhost:8001/api/v1/namespaces/{namespace}/services/https:{namespace}:/proxy/"

        print(f"Copy/pasting the token below and opening the URL:\n\n\t{token}\n\n\t{url}\n")

        pyperclip.copy(token)
        webbrowser.open(url)
