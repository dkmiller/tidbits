from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request
from workers import DurableObject, WorkerEntrypoint


class MyDurableObject(DurableObject):
    """
    State (DB)-aware logic goes here, with knowledge of HTTP requests/responses.
    """
    def __init__(self, ctx, env):
        super().__init__(ctx, env)

    def sql(self, query: str) -> Any:
        return self.ctx.storage.sql.exec(query)#.one() # type: ignore



    async def say_hello(self, name: str):
        self.sql("""
CREATE TABLE IF NOT EXISTS users_v1 (
    username TEXT PRIMARY KEY
    ,count INTEGER
);
        """)

        result = self.sql(f"""
INSERT INTO users_v1 (username, count) 
VALUES ('{name}', 1)
ON CONFLICT(username) 
DO UPDATE SET count = users_v1.count + 1
RETURNING count;
""").one()

        return result.count


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
    greeting = await my_obj.say_hello(name)

    return {"message": str(greeting), "name": name}
