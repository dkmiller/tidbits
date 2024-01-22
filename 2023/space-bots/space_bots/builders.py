from typing import Type, TypeVar

import inflect
import streamlit as st
import streamlit_pydantic as sp
from injector import Injector, Module, multiprovider, provider, singleton

from space_bots.actions import Mine, Travel
from space_bots.client import SpacebotsClient
from space_bots.models import Action, Config, Fleet, System, User


@st.cache_data(ttl=60)
def _systems(_client: SpacebotsClient, _fleets: list[Fleet]) -> dict[str, System]:
    queue = [fleet.location_system_id for fleet in _fleets]

    visited = set()
    rv = {}

    while queue:
        current = queue.pop()
        if current in visited:
            continue
        visited.add(current)
        system = _client.system(current)
        rv[current] = system
        queue.extend(system.neighboring_systems)

    return rv


class Builder(Module):
    @multiprovider
    def actions(self, fleets: list[Fleet], systems: dict[str, System], mine: Mine, travel: Travel) -> list[Action]:
        rv: list[Action] = [travel]

        if any(systems[fleet.location_system_id].asteroid for fleet in fleets):
            rv.append(mine)

        return rv

    @singleton
    @provider
    def config(self) -> Config:
        with st.expander("Configuration"):
            return sp.pydantic_form(key=f"config-{id(self)}", model=Config) or Config()

    @provider
    def client(self, config: Config) -> SpacebotsClient:
        return SpacebotsClient(config.token)

    @multiprovider
    def fleets(self, client: SpacebotsClient) -> list[Fleet]:
        return client.fleets()

    @singleton
    @provider
    def user(self, client: SpacebotsClient) -> User:
        return client.user()

    @singleton
    @provider
    def inflect(self) -> inflect.engine:
        return inflect.engine()

    @multiprovider
    def systems(
        self, client: SpacebotsClient, fleets: list[Fleet]
    ) -> dict[str, System]:
        return _systems(client, fleets)


# This breaks things somehow...
# @st.cache_resource
def injector():
    return Injector([Builder()], auto_bind=True)


T = TypeVar("T")


# @st.cache_resource(ttl=60)
def get(type: Type[T]) -> T:
    """
    Cached. If you don't want caching use `injector().get(type)` instead.
    """
    return injector().get(type)
