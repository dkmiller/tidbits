import uvicorn
from fastapi import FastAPI, Request

from src.m import foo
from src.otel import setup as otel_setup

from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

otel_setup()

tracer = trace.get_tracer(__name__)
app = FastAPI()


@app.get("/probe")
def probe(request: Request):
    return {
        "headers": request.headers,
        "foo": foo(),
        "otel": {
            "trace_id": trace.get_current_span().get_span_context().trace_id
        }
    }


FastAPIInstrumentor.instrument_app(app)


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
