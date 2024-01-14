import json

import typer

from space_bots.client import SpacebotsClient

app = typer.Typer(pretty_exceptions_enable=False)


def show(obj):
    print(json.dumps(obj, indent=2))


@app.command()
def fleets(user: str = "my"):
    client = SpacebotsClient()
    fleets = client.fleets(user)
    show(fleets)


@app.command()
def systems(system: str):
    client = SpacebotsClient()
    system_res = client.system(system)
    show(system_res)


@app.command()
def travel(fleet: str, system: str):
    client = SpacebotsClient()
    res = client.travel(fleet, system)
    show(res)

@app.command()
def mine(fleet: str):
    client = SpacebotsClient()
    res = client.mine(fleet)
    show(res)
