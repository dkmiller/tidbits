from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

tracer = trace.get_tracer(__name__)

from config import config, setup

app = FastAPI()


@app.get("/route")
async def route(
    time=config("http://worldtimeapi.org/api/timezone/America/Los_Angeles"),
    ip=config("https://httpbun.com/ip"),
    cat_fact=config("https://catfact.ninja/fact"),
):
    span = trace.get_current_span()
    context = span.get_span_context()

    return {
        "time_config": time,
        "ip_config": ip,
        "cat_config": cat_fact,
        "trace_id": str(context.trace_id),
        "span_id": str(context.span_id),
    }


FastAPIInstrumentor.instrument_app(app)
setup(app)
