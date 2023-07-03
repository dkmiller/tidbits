import subprocess
import sys
from typing import List

from rich.console import Console
from rich.syntax import Syntax

console = Console()


class ShellClient:
    def __call__(self, *args: str) -> str:
        return self.run(list(args))

    def run(self, args: List[str]) -> str:
        """
        TODO: follow https://janakiev.com/blog/python-shell-commands/ and stream output.
        """
        syntax = Syntax(" ".join(args), "bash")
        with console.status(syntax):
            result = subprocess.run(args, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

            if result.returncode != 0:
                # https://johnlekberg.com/blog/2020-04-03-codec-errors.html
                console.print(result.stdout.decode(errors="replace"))
                console.print(result.stderr.decode(errors="replace"))
                console.print(
                    f"[red]💥 The command below exited with status {result.returncode}:[/red]"
                )
                console.print(syntax)
                sys.exit(result.returncode)

        return result.stdout.decode(errors="replace")
