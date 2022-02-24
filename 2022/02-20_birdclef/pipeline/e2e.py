"""
https://github.com/Azure/azureml-examples/blob/135b53562ada00535d6c3d5c9c76a50935df37d9/sdk/jobs/pipelines-with-components/pipeline-dsl-example.ipynb
"""

from azure.identity import DefaultAzureCredential
from azure.ml import dsl, MLClient
from pathlib import Path
from typing import Callable


def load_local_component(folder_name: str) -> Callable:
    # raise Exception(Path(__file__).absolute())
    component_root = Path(__file__).parent.parent.absolute() / "component"
    rv = dsl.load_component(yaml_file=component_root / f"{folder_name}/component.yaml")
    return rv


kaggle_download = load_local_component("kaggle_download")


@dsl.pipeline(
    compute="cpu-cluster",
    description="Basic Pipeline Job with 3 Hello World components",
)
def end_to_end_pipeline():
    kaggle_job = kaggle_download(competition="birdclef-2022")


p = end_to_end_pipeline()

cred = DefaultAzureCredential()
# https://ml.azure.com/compute/V10032G/details?wsid=/subscriptions/79f57c16-00fe-48da-87d4-5192e86cd047/resourcegroups/Alexander256/workspaces/Alexander256V100&tid=72f988bf-86f1-41af-91ab-2d7cd011db47
ml_client = MLClient(
    cred, "79f57c16-00fe-48da-87d4-5192e86cd047", "Alexander256", "Alexander256V100"
)

ml_client.jobs.create_or_update(
    p, experiment_name="kaggle-birdclef", continue_run_on_step_failure=True
)
