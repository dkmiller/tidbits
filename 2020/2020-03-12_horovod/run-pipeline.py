import argparse
import os
from azureml.core import Experiment
from azureml.core.compute import ComputeTarget
from azureml.core.workspace import Workspace
from azureml.train.dnn import PyTorch, Mpi


def main(args, ws):
    compute = ws.compute_targets[args.cluster]
    print(compute.get_status().serialize())
    
    experiment = Experiment(ws, name='pytorch-distributed-horovod')
    estimator = PyTorch(source_directory=os.path.dirname(os.path.abspath(__file__)),
                    compute_target=compute,
                    entry_script='train.py',
                    node_count=args.nodes,
                    distributed_training=Mpi(),
                    use_gpu=True)
    run = experiment.submit(estimator)
    print(run)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster', default='gpucluster')
    parser.add_argument('--nodes', default=2, type=int)

    args = parser.parse_args()
    workspace = Workspace.from_config()
    main(args, workspace)
