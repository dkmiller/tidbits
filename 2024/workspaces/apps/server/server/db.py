from sqlalchemy import Engine
from sqlmodel import SQLModel

from server.injection import get


def create_db_and_tables():
    engine = get(Engine)
    SQLModel.metadata.create_all(engine)
