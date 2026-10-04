import logging
import os
from typing import Literal

from langchain_openai import ChatOpenAI
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI

from story.models import Dotenv

log = logging.getLogger(__name__)

LlmKind = Literal["langchain", "openai-sdk"]


def llm_client(kind: LlmKind, retries: int = 5, timeout: int = 30 * 60) -> ChatOpenAI | AsyncOpenAI:
    dotenv = Dotenv()
    log.debug("Using API endpoint %s", dotenv.api_endpoint)

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "tidbits-story"
    os.environ["LANGSMITH_API_KEY"] = dotenv.langsmith_key

    match kind:
        case "langchain":
            return ChatOpenAI(
                model=dotenv.model,
                api_key=dotenv.api_key,
                base_url=dotenv.api_endpoint,
                max_retries=retries,
                timeout=timeout
            )
        case "openai-sdk":
            rv = AsyncOpenAI(
                api_key=dotenv.api_key,
                base_url=dotenv.api_endpoint,
                max_retries=retries,
                timeout=timeout,
            )

            return wrap_openai(rv)
