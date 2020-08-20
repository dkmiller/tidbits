from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace
from azureml.core.compute import ComputeInstance
from azureml.core.environment import DEFAULT_GPU_IMAGE
from pyconfigurableml.entry import run


def main(config, log):
    ws = Workspace(**config.workspace)

    env = Environment.from_conda_specification(**config.environment)
    env.docker.enabled = True
    env.docker.base_image = DEFAULT_GPU_IMAGE

    src = ScriptRunConfig(**config.run_config)
    src.run_config.environment = env
    src.run_config.target = ws.compute_targets[config.compute]

    experiment = Experiment(workspace=ws, name=config.experiment)
    run = experiment.submit(src)

    aml_url = run.get_portal_url()
    log.info(f'Run URL: {aml_url}')


run(main, __file__, __name__)
