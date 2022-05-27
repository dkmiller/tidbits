from azure.identity import DefaultAzureCredential
from azure.ai.ml import dsl, MLClient, load_component


download_reddit = load_component(
    path="./components/download_reddit_data/component.yaml"
)


@dsl.pipeline(name="dpv2", default_compute="cpu-cluster")
def sample_pipeline():
    reddit_step = download_reddit(
        client_id="XnkMHEYUujv1wA7EkmToWg",
        client_secret="secret://reddit-client-secret",
        subreddits="news,funny",
    )
    reddit_step.environment_variables = {"AZUREML_COMPUTE_USE_COMMON_RUNTIME": "true"}


p_job = sample_pipeline()


cred = DefaultAzureCredential()
ml_client = MLClient(
    cred, "48bbc269-ce89-4f6f-9a12-c6f91fcb772d", "aml1p-rg", "aml1p-ml-wus2"
)

job = ml_client.jobs.create_or_update(p_job, experiment_name="Huggingface-Reddit")

job_url = job.services["Studio"].endpoint
print(job_url)
