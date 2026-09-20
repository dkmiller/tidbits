from workers import WorkerEntrypoint

from app import app
from durable import MyDurableObject as MyDurableObject  # noqa: PLC0414


class Default(WorkerEntrypoint):
    """
    "Pass through" between FastAPI's ASGI application and the CloudFlare
    Worker.
    """

    async def fetch(self, request):
        import asgi

        return await asgi.fetch(app, request.js_object, self.env)  # type: ignore
