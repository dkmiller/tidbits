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
        timeout=30 * 60,  # Thirty minutes.
    )

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "tidbits-story"
    os.environ["LANGSMITH_API_KEY"] = dotenv.langsmith_key
    return wrap_openai(rv)


@dataclass
class Ai:
    model: str = field(default_factory=lambda: Dotenv().model)
    _openai_client: AsyncOpenAI = field(default_factory=llm_factory)

    async def llm(self, format: type[T], system: str, user: str) -> T:
        log.debug("Calling %s with %s :: %s", self.model, system, user)
        response = await self._openai_client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=format,
        )
        log.debug(response)
        return response.choices[0].message.parsed

    async def novella(
        self, system: str, setting, prompt: str, breakdown: str, style: str
    ) -> str:
        story_setup = await self.llm(
            StorySetup,
            system,
            f"""Use this as inspiration:

> {prompt}

Follow the inspiration above and the guide below to generate the title and setup for a novella:

{setting}""",
        )
        log.info(
            "Generated story setup:# %s\n\n%s", story_setup.title, story_setup.setting
        )

        chapter_guides = await self.llm(
            Story,
            system,
            f"""You are provided with a novella setup and authoring guide below.
Follow them to generate a chapter breakdown for the novella.

## Setup

{story_setup}

## Authoring guide

{breakdown}
""",
        )
        log.info("Generated guides for %s chapters", len(chapter_guides.chapters))
        for index, chapter in enumerate(chapter_guides.chapters):
            log.info("Chapter %s:\n\n%s", index, chapter)

        chapters = await asyncio.gather(
            *[
                self.llm(
                    Chapter,
                    system,
                    f"""You are provided with a novella setup, style guide, and chapter
prompt below. Follow them to generate the chapter in full: 4000 words or so.

## Novella setup

{story_setup}

## Style guide

{style}

## Chapter prompt

{guide}

---

Follow the novella setup, style guide, and chapter prompt to generate a title + content for
the chapter.
""",
                )
                for guide in chapter_guides.chapters
            ]
        )
        log.info("Generated chapters")

        rv = f"# {story_setup.title}\n\n" + "\n".join(
            [f"## {chapter.title}\n\n{chapter.content}" for chapter in chapters]
        )
        return rv
