import joblib
import json
import numpy as np
from azureml.core.model import Model


def init():
    global MODEL
    model_path = Model.get_model_path(model_name='safe_driver_prediction')
    MODEL = joblib.load(model_path)

def run(raw_data, request_headers):
    global MODEL
    data = json.loads(raw_data)['data']
    data = np.array(data)
    result = MODEL.predict(data)

    return {"result": result.tolist()}
