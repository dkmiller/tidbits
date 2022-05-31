from azure.identity import DefaultAzureCredential
from azure.ai.ml import dsl, MLClient, load_component
from dataclasses import dataclass
from functools import cached_property
import hydra
import logging


log = logging.getLogger(__name__)


@dataclass
class HuggingfaceReddit:
    config: object

    def component(self, key: str):
        path = f"./components/{key}/component.yaml"
        rv = load_component(path=path)
        return rv

    @property
    def default_compute(self) -> str:
        return self.config.compute.default

    @property
    def gpu_compute(self) -> str:
        return self.config.compute.gpu

    @cached_property
    def ml_client(self):
        cred = DefaultAzureCredential()
        return MLClient(cred, **self.config.aml)

    @property
    def pipeline(self):
        download_reddit = self.component("download_reddit_data")
        prepare_json = self.component("prepare_json_data")
        huggingface = self.component("finetune_huggingface")

        @dsl.pipeline(
            default_compute=self.default_compute, continue_on_step_failure=True
        )
        def huggingface_reddit(
            batch_size: int,
            client_id: str,
            client_secret: str,
            model: str,
            post_limit: int,
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
                post_limit=post_limit,
                posts_per_file=posts_per_file,
                subreddits=subreddits,
            )
            reddit_step.environment_variables = {
                "AZUREML_COMPUTE_USE_COMMON_RUNTIME": "true"
            }

            json_step = prepare_json(
                input_directory=reddit_step.outputs.output_data,
                output_file_name=output_file_name,
                source_jsonpaths=source_jsonpaths,
                source_key=source_key,
                target_jsonpath=target_jsonpath,
                target_key=target_key,
            )
            json_step.environment_variables = {
                "AZUREML_COMPUTE_USE_COMMON_RUNTIME": "true"
            }

            train_step = huggingface(
                batch_size=batch_size,
                model=model,
                train_data=json_step.outputs.output_data,
                train_file=output_file_name,
                num_epochs=50,
            )
            # https://github.com/Azure/azureml-examples/blob/main/sdk/jobs/pipelines/2b_train_cifar_10_with_pytorch/train_cifar_10_with_pytorch.ipynb
            # Sadly, can't set pipeline parameter to compute.
            train_step.compute = self.gpu_compute
            train_step.environment_variables = {
                "AZUREML_COMPUTE_USE_COMMON_RUNTIME": "true",
            }

        return huggingface_reddit

    def submit(self):
        pipeline = self.pipeline(**self.config.pipeline)
        log.info(f"Created pipeline {pipeline}")

        job = self.ml_client.jobs.create_or_update(
            pipeline, experiment_name=self.config.experiment.name
        )

        job_url = job.services["Studio"].endpoint
        log.info(f"Submitted {job_url}")
        return job


@hydra.main(config_name="huggingface_reddit", config_path=".", version_base=None)
def main(config):
    for k, level in config.logging.items():
        logging.getLogger(k).setLevel(level)

    huggingface_reddit = HuggingfaceReddit(config)
    huggingface_reddit.submit()


if __name__ == "__main__":
    main()
