from pathlib import Path
from typing import Any

from workers import DurableObject


class MyDurableObject(DurableObject):
    """
    State (DB)-aware logic goes here, with knowledge of HTTP requests/responses.
    """

    def __init__(self, ctx, env):
        super().__init__(ctx, env)

        self.sql("creation.sql")

    def sql(self, template: str, *parameters) -> Any:
        """
        Supports either a path inside ./sql/ or an inline (raw) SQL query.
        """
        try:
            path = Path(__file__).parent / "sql" / template
            query = path.read_text()
        except:  # noqa: E722
            query = template
        return self.ctx.storage.sql.exec(query, *parameters)  # type: ignore

    async def say_hello(self, name: str):
        result = self.sql("increment.sql", name)

        return result.one().count
