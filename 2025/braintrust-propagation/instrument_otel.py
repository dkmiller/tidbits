from braintrust import Eval, init_dataset
from opentelemetry import trace

from metrics import precision_recall_score
from otel import opentelemetry_init


tracer = trace.get_tracer(__name__)


@tracer.start_as_current_span("fake_agent")
def fake_agent(input: str) -> list[str]:
    return ["1",]
    


def dummy_task(input: str) -> list[str]:
    return fake_agent(input)


PROJECT_ID = "9a892c95-d81b-43d7-952b-13a3596d9599"

opentelemetry_init(PROJECT_ID)

Eval(
    PROJECT_ID,
    # Fails if project ID here; only project name works.
    data=init_dataset(project="project-63b5607c", name="help_article_retrieval_generated_v3"),
    task=dummy_task,
    scores=[precision_recall_score,],
    experiment_name="Test with dummy task",
)
