import argparse
import os
from azureml.core import Experiment, Workspace
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice


def main(args, ws):
    dir = os.path.dirname(os.path.abspath(__file__))
    ic = InferenceConfig(runtime='python',
                         source_directory=dir,
                         entry_script='score.py',
                         conda_file='environment.yml')

    dc = AciWebservice.deploy_configuration(cpu_cores=args.cores, memory_gb=args.memory)
    m = ws.models[args.model_name]

    service = Model.deploy(ws, args.service_name, [m], ic, dc)
    service.wait_for_deployment(True)
    print(service.state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cores', default=1, type=int)
    parser.add_argument('--memory', default=1, type=int)
    parser.add_argument('--model_name', default='safe_driver_prediction')
    parser.add_argument('--service_name', default='danmill-safe-driver-2')

    args = parser.parse_args()
    ws = Workspace.from_config()

    main(args, ws)
