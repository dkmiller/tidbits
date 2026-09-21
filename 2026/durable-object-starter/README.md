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
