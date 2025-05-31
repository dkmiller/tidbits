import logging
from dotenv import dotenv_values
from strands import Agent, tool
from strands.models.openai import OpenAIModel
from strands_tools import calculator, current_time, python_repl
from sympy import prime

logging.getLogger("strands").setLevel(logging.DEBUG)


model = OpenAIModel(
    client_args={
        "api_key": dotenv_values()["openai_api_key"],
    },
    model_id="gpt-4o",
    params={"max_tokens": 1000}
)



@tool
def nth_prime(index: int) -> int:
    """
    Calculate the Nth prime number, e.g. 1 -> 2, 2 -> 3, 3 -> 5, ....
    """
    # https://stackoverflow.com/a/42440056
    return prime(nth=index)


agent = Agent(model=model, tools=[calculator, current_time, python_repl, nth_prime])


# Ask the agent a question that uses the available tools
message = """
What is the 30th prime number divided by the 10th prime number. Use only the tools provided.

Output a script that does what we just spoke about! Use your python tools to confirm that the script
works before outputting it
"""
agent(message)
