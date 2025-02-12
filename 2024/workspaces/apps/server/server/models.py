from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class Workspace(SQLModel, table=True):
    id: str = Field(default=None, primary_key=True)
    # https://sqlmodel.tiangolo.com/tutorial/connect/create-connected-tables/#create-the-tables
    variant: str = Field(foreign_key="variant.name")


class WorkspaceResponse(Workspace, table=False):
    status: str


class Variant(SQLModel, table=True):
    """
    Jinja-templated assuming `port` (and later, `path` or `user`).
    """

    name: str = Field(primary_key=True)
    # https://stackoverflow.com/a/79368918
    container_args: list[str] = Field(sa_column=Column(JSON))
    docker_image: str = Field()
    ports: list[int] = Field(sa_column=Column(JSON))
    readiness: str = Field()
    """
    Implicitly on the FIRST port.
    """
