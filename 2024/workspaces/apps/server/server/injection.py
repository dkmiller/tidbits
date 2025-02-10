import asyncio
import atexit

from fastapi_injector import RequestScopeOptions, attach_injector
from injector import Injector, Module, provider, singleton
from kubernetes_asyncio.client import ApiClient
from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine


def exit_handler(api: ApiClient):
    asyncio.run(api.close())


class Builder(Module):
    @provider
    def api_client(self) -> ApiClient:
        # TODO: get request-scoped cleanup working.
        # https://github.com/matyasrichter/fastapi-injector?tab=readme-ov-file#dependency-cleanup
        rv = ApiClient()
        atexit.register(exit_handler, (rv))
        return rv

    @singleton
    @provider
    def engine(self) -> Engine:
        # Ensure models are picked up.
        import server.models as _  # noqa: F401

        sqlite_file_name = "workspaces.db"
        sqlite_url = f"sqlite:///{sqlite_file_name}"
        connect_args = {"check_same_thread": False}
        rv = create_engine(sqlite_url, connect_args=connect_args)
        SQLModel.metadata.create_all(rv)
        return rv

    @provider
    def session(self, engine: Engine) -> Session:
        return Session(engine)


def injector():
    return Injector([Builder()], auto_bind=True)


def attach(app):
    attach_injector(app, injector(), options=RequestScopeOptions(enable_cleanup=True))
