import logging
import os

from braintrust import Eval, init_dataset, traced

from metrics import precision_recall_score

logging.basicConfig(level="DEBUG")


@traced
def fake_agent(input: str) -> list[str]:
    return ["1"]


def dummy_task(input: str) -> list[str]:
    return fake_agent(input)


Eval(
    os.environ["BRAINTRUST_PROJECT_NAME"],
    data=init_dataset(
        project=os.environ["BRAINTRUST_PROJECT_NAME"],
        name=os.environ["BRAINTRUST_DATASET_NAME"],
    ),
    task=dummy_task,
    scores=[precision_recall_score],
    experiment_name="Test with dummy task",
)
