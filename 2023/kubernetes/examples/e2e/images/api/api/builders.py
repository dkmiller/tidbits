from socket import AF_INET

from aiohttp import ClientSession, ClientTimeout, TCPConnector
from injector import Injector, Module, provider, singleton
from opentelemetry import trace

tracer = trace.get_tracer(__name__)


class Builder(Module):
    @singleton
    @provider
    @tracer.start_as_current_span("client_session")
    def client_session(self) -> ClientSession:
        timeout = ClientTimeout(total=2)
        connector = TCPConnector(family=AF_INET, limit_per_host=100)
        return ClientSession(timeout=timeout, connector=connector)


@tracer.start_as_current_span("injector")
def injector() -> Injector:
    return Injector([Builder()], auto_bind=True)
