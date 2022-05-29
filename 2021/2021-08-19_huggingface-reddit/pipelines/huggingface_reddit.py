from azure.identity import DefaultAzureCredential
from azure.ai.ml import dsl, MLClient, load_component
import hydra
import logging


log = logging.getLogger(__name__)


download_reddit = load_component(
    path="./components/download_reddit_data/component.yaml"
)
prepare_json = load_component(path="./components/prepare_json_data/component.yaml")
huggingface = load_component(path="./components/finetune_huggingface/component.yaml")


# https://docs.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml.dsl?view=azure-ml-py
@dsl.pipeline(default_compute="cpu-cluster", continue_on_step_failure=True)
def sample_pipeline(
    client_id: str,
    client_secret: str,
    subreddits: str,
    output_file_name: str,
    posts_per_file: int,
    source_jsonpaths: str,
    source_key: str,
    target_jsonpath: str,
    target_key: str,
):
    reddit_step = download_reddit(
        client_id=client_id,
        client_secret=client_secret,
        posts_per_file=posts_per_file,
        subreddits=subreddits,
    )
    reddit_step.environment_variables = {"AZUREML_COMPUTE_USE_COMMON_RUNTIME": "true"}

    json_step = prepare_json(
        input_directory=reddit_step.outputs.output_data,
        output_file_name=output_file_name,
        source_jsonpaths=source_jsonpaths,
        source_key=source_key,
        target_jsonpath=target_jsonpath,
        target_key=target_key,
    )
    json_step.environment_variables = {"AZUREML_COMPUTE_USE_COMMON_RUNTIME": "true"}

    train_step = huggingface(
        train_data=json_step.outputs.output_data, train_file=output_file_name
    )
    # https://github.com/Azure/azureml-examples/blob/main/sdk/jobs/pipelines/2b_train_cifar_10_with_pytorch/train_cifar_10_with_pytorch.ipynb
    train_step.compute = "gpu-cluster"
    train_step.environment_variables = {"AZUREML_COMPUTE_USE_COMMON_RUNTIME": "true"}


@hydra.main(config_name="huggingface_reddit", config_path=".", version_base=None)
def main(config):
    for k, level in config.logging.items():
        logging.getLogger(k).setLevel(level)

    pipeline = sample_pipeline(**config.pipeline)
    log.info(f"Created pipeline {pipeline}")

    cred = DefaultAzureCredential()
    ml_client = MLClient(cred, **config.aml)

    job = ml_client.jobs.create_or_update(
        pipeline, experiment_name="Huggingface-Reddit"
    )

    job_url = job.services["Studio"].endpoint
    log.info(f"Submitted {job_url}")


if __name__ == "__main__":
    main()
