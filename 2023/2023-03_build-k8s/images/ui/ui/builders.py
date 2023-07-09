import os

from injector import Injector, Module, provider, singleton
from ui.models import Context


class Builder(Module):
    @singleton
    @provider
    def context(self) -> Context:
        # https://stackoverflow.com/a/54130803
        if "KUBERNETES_SERVICE_HOST" in os.environ:
            return Context.DEPLOYED
        return Context.INTERACTIVE


def injector() -> Injector:
    return Injector([Builder()], auto_bind=True)
