import os
from pathlib import Path

from fastapi import FastAPI, Request

app = FastAPI()


@app.get("/probe")
def probe(request: Request):
    return {
        "environ": os.environ,
        "filesystem": [str(p) for p in Path(__file__).parent.rglob("*")],
        "headers": request.headers,
        "url": str(request.url),
    }
