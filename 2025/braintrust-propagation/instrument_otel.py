import os
from braintrust import Eval, init_dataset, init
from opentelemetry import trace

from metrics import precision_recall_score
from otel import opentelemetry_init

tracer = trace.get_tracer(__name__)


# Hmm... this is not showing up.
@tracer.start_as_current_span("fake_agent")
def fake_agent(input: str) -> list[str]:
    return ["1"]


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
