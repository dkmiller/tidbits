import uvicorn
from fastapi import FastAPI, Request


app = FastAPI()


@app.post("/items/{rest}")
def read_item(request: Request, rest: str):
    return {"a": "b"}


def main():
    # https://stackoverflow.com/a/62856862

    uvicorn.run(app, host="0.0.0.0", port=8000)
