from strands import tool
from sympy import prime


@tool
def nth_prime(index: int) -> int:
    """
    Calculate the Nth prime number, e.g. 1 -> 2, 2 -> 3, 3 -> 5, ....
    """
    # https://stackoverflow.com/a/42440056
    return prime(nth=index)
