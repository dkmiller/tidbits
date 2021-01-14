"""
Submit job the "old school" way, using the Python SDK.

For reference, see:
https://azure.github.io/azureml-web/docs/cheatsheet/script-run-config
"""


from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace


ws = Workspace("48bbc269-ce89-4f6f-9a12-c6f91fcb772d", "aml1p-rg", "aml1p-ml-wus2")

env = Environment.from_conda_specification("hydra-pl", "environment.yml")
env.docker.base_image = (
    "mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.2-cudnn7-ubuntu18.04"
)

config = ScriptRunConfig(
    source_directory=".",
    script="train.py",
    compute_target="gpu-nc12-lowpri",
    environment=env,
    arguments=["trainer.gpus=1", "trainer.num_nodes=1"],
)

exp = Experiment(ws, "azuremlv2")

run = exp.submit(config, tags={"from": "sdkv1"})

print(run.get_portal_url())
