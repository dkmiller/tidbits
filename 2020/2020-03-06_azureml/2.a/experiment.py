import argparse
import os
from azureml.core import Experiment, Workspace
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import EstimatorStep
from azureml.train.estimator import Estimator


def estimator(data, store, compute):
    estimator = Estimator(source_directory=os.path.dirname(os.path.abspath(__file__)),
                          compute_target=compute,
                          entry_script='train.py',

                          pip_packages=['azureml-dataprep', 'lightgbm'])

    output = PipelineData("output", datastore=store)

    step = EstimatorStep(name=os.path.basename(__file__),
                         estimator=estimator,
                         estimator_entry_script_arguments=[
                             '--input_dir', data,
                             '--output_dir', output],
                         inputs=[data],
                         outputs=[output],
                         compute_target=estimator._compute_target,
                         allow_reuse=True)

    return step, output


def main(args, ws):
    print(ws)
    compute = ws.compute_targets[args.compute]
    datastore = ws.get_default_datastore()
    name = 'porto_seguro_safe_driver_prediction_train'
    dataset = ws.datasets[name].to_csv_files().as_named_input(
        name).as_download(f'/tmp/{name}')

    step, _ = estimator(dataset, datastore, compute)
    pipeline = Pipeline(workspace=ws, steps=[step])
    Experiment(ws, 'danmill-azureml-challenge').submit(pipeline)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute', default='cpucluster')

    args = parser.parse_args()
    ws = Workspace.from_config()
    main(args, ws)
