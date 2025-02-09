from contextlib import asynccontextmanager

from fastapi import FastAPI

from server.db import create_db_and_tables
from server.injection import attach
from server.routes import well_known, workspaces


# https://fastapi.tiangolo.com/advanced/events/#lifespan-function
@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield


app = FastAPI(debug=True, lifespan=lifespan)
attach(app)


# https://fastapi.tiangolo.com/tutorial/bigger-applications/#apirouter
app.include_router(well_known.router)
app.include_router(workspaces.router, prefix="/workspaces", tags=["workspaces"])
