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

    def grafana(self, namespace: str):
        password = self.shell("kubect", "get", "secret", "--namespace", namespace, "grafana", "-o", "jsonpath='{.data.admin-password}")
        url = "http://localhost:3000"

        print(f"Copy/pasting the password below and opening the URL:\n\n\t{password}\n\n\t{url}\n")

        pyperclip.copy(password)
        webbrowser.open(url)


# 1. Get your 'admin' user password by running:

#    kubectl get secret --namespace default grafana -o jsonpath="{.data.admin-password}" | base64 --decode ; echo


# 2. The Grafana server can be accessed via port 80 on the following DNS name from within your cluster:

#    grafana.default.svc.cluster.local

#    Get the Grafana URL to visit by running these commands in the same shell:
#      export POD_NAME=$(kubectl get pods --namespace default -l "app.kubernetes.io/name=grafana,app.kubernetes.io/instance=grafana" -o jsonpath="{.items[0].metadata.name}")
#      kubectl --namespace default port-forward $POD_NAME 3000

# 3. Login with the password from step 1 and the username: admin
# #################################################################################
# ######   WARNING: Persistence is disabled!!! You will lose your data when   #####
# ######            the Grafana pod is terminated.                            #####
# #################################################################################