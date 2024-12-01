from typing import Type, TypeVar

from injector import Injector, Module, provider
from fastapi_injector import attach_injector, RequestScopeOptions

from sqlmodel import Session, create_engine
from sqlalchemy import Engine

T = TypeVar("T")


class Builder(Module):
    @provider
    def engine(self) -> Engine:
        import models as _ # Ensure models are picked up...

        sqlite_file_name = "workspaces.db"
        sqlite_url = f"sqlite:///{sqlite_file_name}"
        connect_args = {"check_same_thread": False}
        return create_engine(sqlite_url, connect_args=connect_args)

    @provider
    def session(self, engine: Engine) -> Session:
        return Session(engine)
    

def injector():
    return Injector([Builder()], auto_bind=True)
    

def attach(app):
    attach_injector(app, injector(), options=RequestScopeOptions(enable_cleanup=True))


def get(interface: Type[T]) -> T:
    return injector().get(interface)
