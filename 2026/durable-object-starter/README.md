## Usage

https://developers.cloudflare.com/durable-objects/get-started/

You can run the Worker defined by your new project by executing `wrangler dev` in this
directory. This will start up an HTTP server and will allow you to iterate on your
Worker without having to restart `wrangler`.

### Types and autocomplete

This project also includes a pyproject.toml file with some requirements which
set up autocomplete and type hints for this Python Workers project.

To get these installed you'll need `uv`, which you can install by following
https://docs.astral.sh/uv/getting-started/installation/.

``` bash
pip install uv

pywrangler sync

# ...
uv venv
uv sync
```

Then point your editor's Python plugin at the `.venv` directory. You should then have working
autocomplete and type information in your editor.

...

``` bash
npx wrangler dev

# Then...
curl localhost:8787/hi/dan

# Deploy...
npx wrangler deploy
```

## Links

- https://fastapi.tiangolo.com/tutorial/dependencies/#create-a-dependency-or-dependable
- https://github.com/cloudflare/python-workers-examples/blob/main/fastapi/src/worker.py
- https://dbfiddle.dev/sqlite
- https://durable-object-starter.dkmiller.workers.dev/docs
