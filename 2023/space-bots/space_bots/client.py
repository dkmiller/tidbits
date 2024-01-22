import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests

from space_bots.models import Fleet, System, User

StrDict = Dict[str, Any]
"""
Dictionary with string-typed keys.
"""


@dataclass(frozen=True)
class SpacebotsClient:
    """
    https://space-bots.longwelwind.net/docs/
    """

    token: str = field(default_factory=lambda: os.environ["SPACEBOTS_TOKEN"])
    endpoint: str = "https://space-bots.longwelwind.net/v1"

    def _request(
        self, method: str, route: str, *, json: Optional[StrDict] = None
    ) -> str:
        url = f"{self.endpoint}/{route}"
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.request(method, url, headers=headers, json=json)
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code} {response.text}")
        return response.text

    def request(
        self, method: str, route: str, *, json: Optional[StrDict] = None
    ) -> StrDict:
        url = f"{self.endpoint}/{route}"
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.request(method, url, headers=headers, json=json)
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code} {response.text}")
        return response.json()

    def buy_ships(self, fleet: str, ships: dict[str, int]):
        return self.request(
            "POST", f"fleets/{fleet}/buy-ships", json={"shipsToBuy": ships}
        )

    def fleets(self, user: str = "my") -> list[Fleet]:
        # It's annoying that "my" here and "me" below are different.
        response = self.request("GET", f"fleets/{user}")  # type: ignore
        return [Fleet.from_dict(fleet) for fleet in response]  # type: ignore

    def market(self, system: str, resource: str):
        return self.request("GET", f"systems/{system}/market/resources/{resource}")

    def system(self, id: str) -> System:
        response = self.request("GET", f"systems/{id}")
        return System(
            id=response["id"],
            name=response["name"],
            x=response["x"],
            y=response["y"],
            neighboring_systems=[s["systemId"] for s in response["neighboringSystems"]],
            asteroid=response.get("asteroid", {}),
        )

    def travel(self, fleet: str, system: str):
        return self.request(
            "POST", f"fleets/{fleet}/travel", json={"destinationSystemId": system}
        )

    def mine(self, fleet: str):
        return self.request("POST", f"fleets/{fleet}/mine")

    def sell(self, fleet: str, resources: dict[str, int]):
        return self.request(
            "POST", f"fleets/{fleet}/direct-sell", json={"resources": resources}
        )

    def ship_types(self) -> list[StrDict]:
        return self.request("GET", "ship-types")  # type: ignore

    def transfer(
        self, source: str, ships: dict[str, int], target: Optional[str] = None
    ):
        body = {"shipsFromFleetToTarget": ships}
        if target:
            body["targetFleetId"] = target
        return self.request("POST", f"fleets/{source}/transfer", json=body)

    def user(self, user: str = "me") -> User:
        response = self.request("GET", f"users/{user}")
        return User(
            id=response["id"],
            name=response["name"],
            credits=response["credits"],
            created_at=response["createdAt"],
            registered=response["registered"],
        )
