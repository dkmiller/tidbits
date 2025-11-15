from injector import Injector
from strands import Agent

from src.builder import Builder


def main():
    injector = Injector(Builder())
    agent = injector.get(Agent)
    response = agent(prompt="What do you know?")
    print(response)


if __name__ == "__main__":
    main()
