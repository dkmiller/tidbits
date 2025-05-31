from mcp.server import FastMCP

mcp = FastMCP("Mathematics Server")

@mcp.tool(description="Add two numbers together")
def add(x: int, y: int) -> int:
    """Add two numbers and return the result."""
    return x + y

# mcp.run(transport="streamable-http")
