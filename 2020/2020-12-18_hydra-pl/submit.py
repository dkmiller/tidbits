"""
Submit job the "old school" way, using the Python SDK.

For reference, see:
https://azure.github.io/azureml-web/docs/cheatsheet/script-run-config
"""


from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace
from azureml.core.runconfig import MpiConfiguration


ws = Workspace("48bbc269-ce89-4f6f-9a12-c6f91fcb772d", "aml1p-rg", "aml1p-ml-wus2")

env = Environment.from_conda_specification("hydra-pl", "environment.yml")
env.docker.enabled = True
env.docker.base_image = (
    "mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.2-cudnn8-ubuntu18.04"
)

# ==============================================================================
node_count = 2
gpus_per_node = -1
cluster = "gpu-nc24-lowpri"
# ==============================================================================

mpi_config = MpiConfiguration(process_count_per_node=1, node_count=node_count)

config = ScriptRunConfig(
    source_directory=".",
    script="train.py",
    compute_target=cluster,
    distributed_job_config=mpi_config,
    environment=env,
    arguments=[
        f"trainer.gpus={gpus_per_node}",
        f"trainer.num_nodes={node_count}",
        "+trainer.accelerator=ddp",
    ],
)

exp = Experiment(ws, "azuremlv2")

run = exp.submit(config, tags={"from": "sdkv1"})

print(run.get_portal_url())
