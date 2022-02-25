"""
https://github.com/Azure/azureml-examples/blob/135b53562ada00535d6c3d5c9c76a50935df37d9/sdk/jobs/pipelines-with-components/pipeline-dsl-example.ipynb
"""


from azure.ml import dsl
from core import load_local_component, get_ml_client


@dsl.pipeline(
    compute="cpu-cluster",
    description="Basic Pipeline Job with 3 Hello World components",
)
def end_to_end_pipeline():
    kaggle_download = load_local_component("kaggle_download")

    kaggle_job = kaggle_download(
        competition="birdclef-2022",
        user="antifragilista",
        api_key_vault_name="aml-ds-kv",
        api_key_secret_name="danmill-kaggle-api-key",
    )


def main():
    p = end_to_end_pipeline()
    ml_client = get_ml_client("aml1p-ml-wus2")
    job = ml_client.jobs.create_or_update(
        p, experiment_name="kaggle-birdclef", continue_run_on_step_failure=True
    )
    workspace_url = job.services["Studio"].endpoint
    print(f"Created job: {workspace_url}")


if __name__ == "__main__":
    main()
