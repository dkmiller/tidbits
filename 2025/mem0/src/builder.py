from pathlib import Path

import yaml
from dotenv import dotenv_values
from injector import Module, provider, singleton
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
        # TODO: OmegaConfig for config validation?
        return Config(**(raw | env))

    @provider
    def agent(self, config: Config) -> Agent:
        model = OpenAIModel(client_args={"api_key": config.openai_api_key}, model_id=config.model_id)
        rv = Agent(
            model=model,
            system_prompt=config.system_prompt,
        )
        return rv
