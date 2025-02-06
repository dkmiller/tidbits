from sqlmodel import Field, SQLModel


class Workspace(SQLModel, table=True):
    id: str = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    image_alias: str = Field()
    port: int = Field(8888)


class WorkspaceResponse(Workspace, table=False):
    status: str
