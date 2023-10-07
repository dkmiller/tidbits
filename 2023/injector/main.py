from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from config import config, setup

app = FastAPI()


@app.get("/route")
async def route(time=config("http://worldtimeapi.org/api/timezone/America/Los_Angeles")):
    return {"time_config": time}


FastAPIInstrumentor.instrument_app(app)
setup(app)
