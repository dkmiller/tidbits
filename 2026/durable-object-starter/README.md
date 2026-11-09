# Cloudflare durable workers + FastAPI

Get type hints working...

``` bash
pip install uv
pywrangler sync
```

## Local iteration and deployment

``` bash
npm run dev

# Then...
curl localhost:8787/hi/dan

# Deploy...
npm run deploy

# Then...
curl https://durable-object-starter.dkmiller.workers.dev/hi/dan
```

## Links

- https://developers.cloudflare.com/durable-objects/get-started/
- https://fastapi.tiangolo.com/tutorial/dependencies/#create-a-dependency-or-dependable
- https://github.com/cloudflare/python-workers-examples/blob/main/fastapi/src/worker.py
- https://dbfiddle.dev/sqlite
- https://durable-object-starter.dkmiller.workers.dev/docs
- :x: Not necessary: https://fastapi.tiangolo.com/tutorial/frontend/
- :x: [No SQLAlchemy](https://claude.ai/share/f127c137-bebb-4e7f-a230-30b2326ada68),
  it requires a connection string / "engine" which is not possible with the
  `sql.exec` API.
- :x: https://github.com/javlondevv/fastapi-passkeys, because it requires
  SQLAlchemy or Redis.

**This has become a mess.** Cloudflare runs into coroutine / serialization
issues when doing anything meaningful with the result of `.sql.exec()`.

https://gist.github.com/samuelcolvin/3ff019aa738aa558a185c4fb002b5751
