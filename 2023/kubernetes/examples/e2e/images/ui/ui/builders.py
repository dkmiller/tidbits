import os

from injector import Injector, Module, provider, singleton
from opentelemetry import trace

from ui.models import Context

tracer = trace.get_tracer(__name__)


class Builder(Module):
    @singleton
    @provider
    @tracer.start_as_current_span("context")
    def context(self) -> Context:
        # https://stackoverflow.com/a/54130803
        if "KUBERNETES_SERVICE_HOST" in os.environ:
            return Context.DEPLOYED
        return Context.INTERACTIVE


@tracer.start_as_current_span("injector")
def injector() -> Injector:
    return Injector([Builder()], auto_bind=True)
