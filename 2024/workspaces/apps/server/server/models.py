from pydantic import BaseModel
from sqlmodel import Field, SQLModel


class Workspace(SQLModel, table=True):
    id: str = Field(default=None, primary_key=True)
    variant: str = Field()


class WorkspaceResponse(Workspace, table=False):
    status: str


class Variant(BaseModel):
    """
    Jinja-templated assuming `port` (and later, `path` or `user`).
    """

    container_args: list[str]
    docker_image: str
    name: str
    ports: list[int]
    readiness: str
    """
    Implicitly on the FIRST port.
    """
