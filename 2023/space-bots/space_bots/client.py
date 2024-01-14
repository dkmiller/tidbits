import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests

StrDict = Dict[str, Any]
"""
Dictionary with string-typed keys.
"""


@dataclass
class SpacebotsClient:
    """
    https://space-bots.longwelwind.net/docs/
    """

    token: str = field(default_factory=lambda: os.environ["SPACEBOTS_TOKEN"])
    endpoint: str = "https://space-bots.longwelwind.net/v1"

    def request(
        self, method: str, route: str, *, json: Optional[StrDict] = None
    ) -> StrDict:
        url = f"{self.endpoint}/{route}"
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.request(method, url, headers=headers, json=json)
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code} {response.text}")
        return response.json()

    def fleets(self, user: str):
        return self.request("GET", f"fleets/{user}")

    def system(self, system_id: str):
        return self.request("GET", f"systems/{system_id}")

    def travel(self, fleet: str, system: str):
        return self.request(
            "POST", f"fleets/{fleet}/travel", json={"destinationSystemId": system}
        )

    def mine(self, fleet: str):
        return self.request("POST", f"fleets/{fleet}/mine")
