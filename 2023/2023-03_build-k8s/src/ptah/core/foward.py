import json
from dataclasses import dataclass
from typing import List

from injector import inject
from jsonpath_ng import parse
from rich import print

from ptah.core.process import ProcessClient
from ptah.core.shell import ShellClient


@dataclass
class Port:
    args: List[str]


@inject
@dataclass
class Forward:
    """
    Manage port-forwarding
    """

    process: ProcessClient
    shell: ShellClient

    # TODO: use dependency injection for this!
    def commands(self) -> List[List[str]]:
        # https://stackoverflow.com/a/56259811
        deployments_raw = self.shell("kubectl", "get", "deployments", "-o", "json")
        deployments = json.loads(deployments_raw)
        # TODO: don't assume container ports and metadata names are in the same order.
        # https://stackoverflow.com/a/30683008/
        port_path = parse("$..containerPort")
        name_path = parse("$..metadata.name")

        ports = [m.value for m in port_path.find(deployments)]
        names = [m.value for m in name_path.find(deployments)]

        rv = [["kubectl", "proxy"]]

        for index, number in enumerate(ports):
            name = names[index]
            rv.append(["kubectl", "port-forward", f"deployment/{name}", f"{number}:{number}"])

        print(rv)
        return rv

    def terminate(self):
        commands = self.commands()
        print(f"Terminating {len(commands)} port-forwarding processes")
        for args in commands:
            self.process.terminate(args)

    def ensure(self):
        commands = self.commands()
        print(f"Ensuring {len(commands)} port-forwarding processes")
        for args in commands:
            self.process.ensure(args)
