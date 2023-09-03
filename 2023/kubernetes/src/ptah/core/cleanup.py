from dataclasses import dataclass

from injector import inject
from rich import print

from ptah.core.shell import ShellClient


@inject
@dataclass
class Cleanup:
    shell: ShellClient

    def cleanup(self, whatif: bool):
        self.kind(whatif)
        self.docker(whatif)

    def docker(self, whatif: bool):
        """
        https://docs.docker.com/engine/reference/commandline/system_prune/
        """
        if whatif:
            print("Skipping Docker cleanup")
        else:
            self.shell("docker", "system", "prune", "-a", "-f", "--volumes")

    def kind(self, whatif: bool):
        """
        https://kind.sigs.k8s.io/docs/user/quick-start/
        """
        clusters = self.shell("kind", "get", "clusters")
        kind_clusters = clusters.splitlines()
        # TODO: humanize 'clusters'.
        print(f"Cleaning up {len(kind_clusters)} kind clusters")
        for cluster in kind_clusters:
            if not whatif:
                self.shell("kind", "delete", "cluster", "--name", cluster)
            else:
                print(f"Skipping deletion of kind cluster '{cluster}'")
