import os
from braintrust import Eval, init_dataset, init, current_span
from opentelemetry import trace

from metrics import precision_recall_score
from otel import opentelemetry_init

tracer = trace.get_tracer(__name__)


# Hmm... this is not showing up.
def fake_agent(input: str) -> list[str]:
    # Create a child span using both Braintrust's current_span and OpenTelemetry
    # This ensures the span appears as a child in Braintrust's UI
    with current_span().start_span(name="bt_fake_agent") as braintrust_span:
        # Also create an OpenTelemetry span for additional telemetry
        with tracer.start_as_current_span("ot_fake_agent") as otel_span:
            otel_span.set_attribute("input", input)
            result = ["this was set with otel"]
            otel_span.set_attribute("output", str(result))

            # Log to the Braintrust span as well
            braintrust_span.log(input="bt_input", output="otel_output")
            return ["agent result"]


def dummy_task(input: str) -> list[str]:
    return fake_agent(input)


exp = init(
    experiment="Test with dummy task", project=os.environ["BRAINTRUST_PROJECT_NAME"]
)
opentelemetry_init(experiment_id=exp.id)


Eval(
    os.environ["BRAINTRUST_PROJECT_NAME"],
    data=init_dataset(
        project=os.environ["BRAINTRUST_PROJECT_NAME"],
        name=os.environ["BRAINTRUST_DATASET_NAME"],
    ),
    task=dummy_task,
    scores=[precision_recall_score],
)
