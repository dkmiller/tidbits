from typing import Any

from workers import DurableObject


class MyDurableObject(DurableObject):
    """
    State (DB)-aware logic goes here, with knowledge of HTTP requests/responses.
    """

    def __init__(self, ctx, env):
        super().__init__(ctx, env)

        self.sql("""
CREATE TABLE IF NOT EXISTS users_v2 (
    username TEXT PRIMARY KEY
    ,count INTEGER
);
        """)

    def sql(self, query: str, *parameters) -> Any:
        return self.ctx.storage.sql.exec(query, *parameters)  # type: ignore

    async def say_hello(self, name: str):
        result = self.sql(
            """
INSERT INTO users_v2 (username, count)
VALUES (?, 1)
ON CONFLICT(username)
DO UPDATE SET count = users_v2.count + 1
RETURNING count;
""",
            name,
        )

        return result.one().count
