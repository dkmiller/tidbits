# Space bots

https://space-bots.longwelwind.net/

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
