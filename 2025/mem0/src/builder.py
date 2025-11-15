import os
from pathlib import Path

import yaml
from braintrust.otel import BraintrustSpanProcessor
from dotenv import dotenv_values, load_dotenv
from injector import Module, provider, singleton
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from strands import Agent
from strands.models.openai import OpenAIModel

from src.models import Config


class Builder(Module):
    @singleton
    @provider
    def configuration(self) -> Config:
        config_path = Path(__file__).parents[1] / "config.yml"
        with config_path.open("r") as file:
            raw = yaml.safe_load(file)
        env = dotenv_values(".env")

        # Side effect: for mem0 tool.
        load_dotenv()

        for name, value in raw["env"].items():
            # Handle checked-in configuration of non-sensitive environment variables.
            os.environ[name] = value

        # TODO: OmegaConfig for config validation?
        return Config(**(raw | env))

    @provider
    def agent(self, config: Config, _: TracerProvider) -> Agent:
        model = OpenAIModel(
            client_args={"api_key": config.OPENAI_API_KEY},
            model_id=config.model_id,
        )

        # Hack: this module looks at environment variables import-time.
        from strands_tools import mem0_memory

        # Egregious hack: this module does not expose a better way to set advanced
        # mem0 configuration.
        mem0_memory.Mem0ServiceClient.DEFAULT_CONFIG["embedder"]["config"][
            "embedding_dims"
        ] = config.embedding_dims

        rv = Agent(
            model=model,
            system_prompt=config.system_prompt,
            tools=[mem0_memory],
            agent_id=config.agent_id,
        )
        return rv

    @provider
    def opentelemetry_provider(self, _: Config) -> TracerProvider:
        provider = TracerProvider()
        trace.set_tracer_provider(provider)

        provider.add_span_processor(BraintrustSpanProcessor())  # type: ignore
        return provider
