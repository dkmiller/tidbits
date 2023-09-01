import logging
from socket import AF_INET
from typing import Optional

import aiohttp

log = logging.getLogger("uvicorn")


class SingletonAiohttp:
    aiohttp_client: Optional[aiohttp.ClientSession] = None

    @classmethod
    def client_session(cls) -> aiohttp.ClientSession:
        if cls.aiohttp_client is None:
            log.info("Creating aiohttp client")
            timeout = aiohttp.ClientTimeout()
            connector = aiohttp.TCPConnector(family=AF_INET, limit_per_host=100)
            cls.aiohttp_client = aiohttp.ClientSession(
                timeout=timeout, connector=connector
            )

        return cls.aiohttp_client

    @classmethod
    async def close_aiohttp_client(cls) -> None:
        if cls.aiohttp_client:
            log.info("Closing aiohttp client")
            await cls.aiohttp_client.close()
            cls.aiohttp_client = None
