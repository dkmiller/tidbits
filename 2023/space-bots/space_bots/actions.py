from dataclasses import dataclass
from injector import inject
from typing import TypeVar
import streamlit as st


from space_bots.models import Action, Fleet, System


T = TypeVar("T")



# mineable_fleets = [fleet for fleet in fleets if all_systems[fleet["locationSystemId"]].get("asteroid")]

# if mineable_fleets:
#     actions += ["Mine"]


@inject
@dataclass
class Mine(Action):
    def act(self, option):
        pass

    @property
    def name(self):
        return "Mine"

    def submenu(self) -> T:
        pass



@inject
@dataclass
class Travel(Action):
    fleets: list[Fleet]
    systems: dict[str, System]

    def act(self, option):
        pass

    @property
    def name(self):
        return "Travel"

    def submenu(self) -> T:
        movement = {}
        for fleet in self.fleets:
            system = self.systems[fleet.location_system_id]
            neighbors = system.neighboring_systems
            status_quo = f"Sit still {system}"
            destination = st.radio(
                f"{fleet.id:.8} &mapsto;",
                [status_quo] + [
                    str(self.systems[ns])
                    for ns in neighbors
                ],
                key=f"destination-{fleet.id}",
                horizontal=True,
            )
            if destination == status_quo:
                continue
            destination_id = next(
                ns
                for ns in neighbors
                if self.systems[ns].name in destination
            )
            movement[fleet.id] = destination_id

        if movement and st.button(f":arrow_right: Travel {movement}"):
            return movement
