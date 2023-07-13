import json
import os
from dataclasses import dataclass
from typing import List

from injector import inject
from jsonpath_ng import parse
from rich import print

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

    shell: ShellClient

    # TODO: use dependency injection for this!
    def ports(self) -> List[Port]:
        # https://stackoverflow.com/a/56259811
        deployments_raw = self.shell("kubectl", "get", "deployments", "-o", "json")
        deployments = json.loads(deployments_raw)
        # TODO: don't assume container ports and metadata names are in the same order.
        # https://stackoverflow.com/a/30683008/
        port_path = parse("$..containerPort")
        name_path = parse("$..metadata.name")

        ports = [m.value for m in port_path.find(deployments)]
        names = [m.value for m in name_path.find(deployments)]

        rv = [Port(["proxy"])]

        for index, number in enumerate(ports):
            name = names[index]
            rv.append(Port(["port-forward", f"deployment/{name}", f"{number}:{number}"]))

        print(rv)
        return rv

        # raise Exception(rv)

    def ensure(self, deployment: str):
        # https://stackoverflow.com/a/37468186/
        deployment_raw = self.shell("kubectl", "get", f"deployment/{deployment}", "-o", "json")
        deployment_ = json.loads(deployment_raw)
        port_path = parse("$..containerPort")
        port = [m.value for m in port_path.find(deployment_)][0]
        self.command(f"kubectl port-forward deployment/{deployment} {port}:{port}\n")

    # TODO: retry.
    def command(self, command: str):
        print(f"Running command\n\n\t{command}")
        os.system(command)
