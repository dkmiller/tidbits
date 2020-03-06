import os
from azureml.core import Dataset, Experiment, Workspace

ws = Workspace.from_config()

DATASET_NAME = 'porto_seguro_safe_driver_prediction_train'

datastore = ws.get_default_datastore()

if not DATASET_NAME in ws.datasets:
    dir = os.path.dirname(os.path.abspath(__file__))
    fname = f'{DATASET_NAME}.csv'
    path = os.path.join(dir, 'dsdevops-oh-files', fname)
    print(f'Uploading file {path}.')

    datastore.upload_files([path])

    ds = Dataset.Tabular.from_delimited_files(path=[(datastore, fname)])
    ds = ds.register(workspace=ws, name=DATASET_NAME, description='https://www.kaggle.com/c/porto-seguro-safe-driver-prediction')
    print(f'Created {ds}')
