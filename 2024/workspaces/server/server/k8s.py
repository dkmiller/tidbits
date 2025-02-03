from kubernetes_asyncio import config
from kubernetes_asyncio.client import ApiClient, CoreV1Api


config.load_incluster_config()


async def v1_api():
    # https://github.com/tomplus/kubernetes_asyncio?tab=readme-ov-file#example
    async with ApiClient() as api:
        # https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-with-yield/#a-database-dependency-with-yield
        yield CoreV1Api(api)
