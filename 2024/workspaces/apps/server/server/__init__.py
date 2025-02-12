from fastapi import FastAPI

from server.injection import attach
from server.routes import variants, well_known, workspaces

app = FastAPI(debug=True)
attach(app)


# https://fastapi.tiangolo.com/tutorial/bigger-applications/#apirouter
app.include_router(well_known.router)
app.include_router(variants.router, prefix="/variants", tags=["variants"])
app.include_router(workspaces.router, prefix="/workspaces", tags=["workspaces"])
