from typing import Annotated

from fastapi import Depends, FastAPI, Request
from workers import DurableObject, WorkerEntrypoint


class MyDurableObject(DurableObject):
    """
    State (DB)-aware logic goes here, with knowledge of HTTP requests/responses.
    """
    def __init__(self, ctx, env):
        super().__init__(ctx, env)

    async def say_hello(self):
        result = self.ctx.storage.sql.exec(
            "SELECT 'Hello, World!' as greeting"
        ).one()

        return result.greeting


class Default(WorkerEntrypoint):
    """
    "Pass through" between FastAPI's ASGI application and the CloudFlare
    Worker.
    """
    async def fetch(self, request):
        import asgi

        return await asgi.fetch(app, request.js_object, self.env) # type: ignore


app = FastAPI()


def my_durable_object(request: Request) -> MyDurableObject:
    env = request.scope["env"]
    stub = env.MY_DURABLE_OBJECT.getByName(request.url.path)
    return stub


# https://fastapi.tiangolo.com/tutorial/dependencies/#create-a-dependency-or-dependable
MyObject = Annotated[MyDurableObject, Depends(my_durable_object)]
"""
Resolve a Durable Object instance for an HTTP request.
"""


@app.get("/hi/{name}")
async def say_hi(name: str, my_obj: MyObject):
    greeting = await my_obj.say_hello()

    return {"message": str(greeting), "name": name}
