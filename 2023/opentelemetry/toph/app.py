from random import randint

from fastapi import FastAPI
# from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor


# from opentelemetry import trace
# from opentelemetry.sdk.resources import Resource
# from opentelemetry.sdk.trace import TracerProvider
# from opentelemetry.sdk.trace.export import BatchSpanProcessor
# from opentelemetry.sdk.trace.export import ConsoleSpanExporter

# provider = TracerProvider()
# processor = BatchSpanProcessor(ConsoleSpanExporter())
# provider.add_span_processor(processor)
# trace.set_tracer_provider(provider)
# tracer = trace.get_tracer(__name__)


app = FastAPI()



def roll_sum(sides, rolls):
    sum = 0
    for r in range(0,rolls):
        result = randint(1,sides)
        sum += result
    return str(sum)

@app.get("/roll")
def roll(sides: int, rolls: int):
    # with tracer.start_as_current_span("server_request"):
    return roll_sum(sides,rolls)

# FastAPIInstrumentor.instrument_app(app)
