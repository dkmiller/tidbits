# Space bots

[![slack](https://img.shields.io/static/v1?label=Discord&message=Space+Bots&color=5865F2&logo=Discord)](https://discord.gg/ATeDzSy2Wu)
[![api](https://img.shields.io/static/v1?label=Swagger&message=Space+bots&color=85EA2D&logo=swagger)](https://space-bots.longwelwind.net/docs/)

https://space-bots.longwelwind.net/

https://www.reddit.com/r/programming/comments/18of0d3/space_bots_an_online_multiplayer_space_game_that/

## Playing

```bash
# Get fleets
spacebots fleets

# Information about a system
spacebots systems omega
```

## Development

```bash
ruff format . && ruff --fix . && isort .
```

## To-do

- [ ] Restructure web codebase with dependency injection and
  better caching story + parallelized state update.

https://github.com/openapi-generators/openapi-python-client

Getting undebuggable "Type error" with simple code:

```python
from space_bots_client import AuthenticatedClient
from space_bots_client.api.fleets import get_fleets_my
import os

client = AuthenticatedClient(
    "https://space-bots.longwelwind.net/v1/", os.environ["SPACEBOTS_TOKEN"]
)

print(get_fleets_my.sync(client=client))
```
