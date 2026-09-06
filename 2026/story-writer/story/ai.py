import asyncio
import logging
from dataclasses import dataclass, field
from typing import TypeVar

from openai import AsyncOpenAI

from story.models import Chapter, Dotenv, Story, StorySetup

log = logging.getLogger(__name__)

T = TypeVar("T")


def llm_factory():
    dotenv = Dotenv()
    log.debug("Using API endpoint %s", dotenv.api_endpoint)

    return AsyncOpenAI(
        api_key=dotenv.api_key,
        base_url=dotenv.api_endpoint,
        timeout=10 * 60,  # Ten minutes.
    )


@dataclass
class Ai:
    model: str = field(default_factory=lambda: Dotenv().model)
    _openai_client: AsyncOpenAI = field(default_factory=llm_factory)

    async def llm(self, format: type[T], messages) -> T:
        log.debug("Calling %s with %s", self.model, messages)
        response = await self._openai_client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=format,
        )
        log.debug(response)
        return response.choices[0].message.parsed

    async def novella(
        self, system: str, setting, prompt: str, breakdown: str, style: str
    ) -> str:
        story_setup = await self.llm(
            StorySetup,
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"Using '{prompt}', follow the guide below to generate the title and setup for a novella:\n\n{setting}",
                },
            ],
        )

        chapter_guides = await self.llm(
            Story,
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"Use the story setup below as context:\n\n '{story_setup}'. Now, follow the guide below to generate a chapter breakdown for a novella:\n\n{breakdown}",
                },
            ],
        )

        chapters = await asyncio.gather(
            *[
                self.llm(
                    Chapter,
                    [
                        {"role": "system", "content": style},
                        {
                            "role": "user",
                            "content": f"Use the story setup below as context:\n\n '{story_setup}'.  Now, follow the guide below to generate a title + content for a specific chapter of the novella:\n\n{guide}",
                        },
                    ],
                )
                for guide in chapter_guides.chapters
            ]
        )

        rv = f"# {story_setup.title}\n\n" + "\n".join(
            [f"## {chapter.title}\n\n{chapter.content}" for chapter in chapters]
        )
        return rv
