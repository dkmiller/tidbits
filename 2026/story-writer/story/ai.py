import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import TypeVar

from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI

from story.models import Chapter, Dotenv, Story, StorySetup

log = logging.getLogger(__name__)

T = TypeVar("T")


def llm_factory():
    dotenv = Dotenv()
    log.debug("Using API endpoint %s", dotenv.api_endpoint)

    rv = AsyncOpenAI(
        api_key=dotenv.api_key,
        base_url=dotenv.api_endpoint,
        max_retries=5,
        timeout=20 * 60,  # Ten minutes.
    )

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "tidbits-story"
    os.environ["LANGSMITH_API_KEY"] = dotenv.langsmith_key
    return wrap_openai(rv)


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
                    "content": f"Use this as inspiration:\n> {prompt} Follow the inspiration above and the guide below to generate the title and setup for a novella:\n\n{setting}",
                },
            ],
        )
        log.info(
            "Generated story setup:# %s\n\n%s", story_setup.title, story_setup.setting
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
        log.info("Generated guides for %s chapters", len(chapter_guides.chapters))
        for index, chapter in enumerate(chapter_guides.chapters):
            log.info("Chapter %s:\n\n%s", index, chapter)

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
        log.info("Generated chapters")

        rv = f"# {story_setup.title}\n\n" + "\n".join(
            [f"## {chapter.title}\n\n{chapter.content}" for chapter in chapters]
        )
        return rv
