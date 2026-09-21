from typing import Annotated

from fastapi import Depends, FastAPI, Request

from durable import MyDurableObject

app = FastAPI()


def my_durable_object(request: Request) -> MyDurableObject:
    env = request.scope["env"]
    stub = env.MY_DURABLE_OBJECT.getByName(request.url.path)
    return stub


MyObject = Annotated[MyDurableObject, Depends(my_durable_object)]
"""
Resolve a Durable Object instance for an HTTP request.
"""


@app.get("/hi/{name}")
async def say_hi(name: str, my_obj: MyObject):
    greeting = await my_obj.say_hello(name)

    return {"message": str(greeting), "name": name}
