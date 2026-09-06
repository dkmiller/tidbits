import logging
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from story.models import Dotenv

log = logging.getLogger(__name__)


def llm_factory():
    dotenv = Dotenv()
    log.debug("Using API endpoint %s", dotenv.api_endpoint)

    return AsyncOpenAI(
        api_key=dotenv.api_key,
        base_url=dotenv.api_endpoint,
    )


@dataclass
class Ai:
    model: str
    llm: AsyncOpenAI = field(default_factory=llm_factory)

    async def hello(self):
        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Explain the concept of quantum computing in one sentence.",
                },
            ],
        )
        print(response)
