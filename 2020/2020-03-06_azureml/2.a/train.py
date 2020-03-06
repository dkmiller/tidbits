import argparse
import glob
import joblib
import lightgbm
import numpy as np
import os
import pandas as pd
from azureml.core.run import Run
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from sklearn import metrics


def load_input(path):
    '''
    Follow https://stackoverflow.com/a/3207973 to list all files in a directory.
    '''
    print(f'Searching {args.input_dir} for files.')

    files = [f for f in listdir(args.input_dir) if isfile(join(args.input_dir, f))]

    df = None
    for f in files:
        path = join(args.input_dir, f)
        print(f'Loading {path}')
        if df is None:
            df = pd.read_csv(path)
        else:
            df = df.append(pd.read_csv(path))

    print(f'Data shape: {df.shape}')

    return df


def preprocess_split(df, validation_size, seed):
    features = df.drop(['target', 'id'], axis=1)
    labels = np.array(df['target'])
    features_train, features_valid, labels_train, labels_valid = train_test_split(features, labels, test_size=validation_size, random_state=seed)

    train_data = lightgbm.Dataset(features_train, label=labels_train)
    valid_data = lightgbm.Dataset(features_valid, label=labels_valid, free_raw_data=False)

    return (train_data, valid_data)


def evaluate_model(model, valid, run):
    predictions = model.predict(valid.data)
    fpr, tpr, _ = metrics.roc_curve(valid.label, predictions)
    tags = {
        'auc': metrics.auc(fpr, tpr)
    }
    for k, v in tags.items():
        run.log(k, v)
    return tags


def train_model(train, valid, args, run):
    parameters = {
        'learning_rate': args.learning_rate,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'sub_feature': args.sub_feature,
        'num_leaves': args.num_leaves,
        'min_data': args.min_data,
        'min_hessian': args.min_hessian,
        'verbose': args.verbose
    }

    # https://stackoverflow.com/a/3294899
    for key, value in parameters.items():
        run.log(key, value)

    model = lightgbm.train(parameters,
                           train,
                           valid_sets=valid,
                           num_boost_round=500,
                           early_stopping_rounds=20)

    return model


def save_model(model, output_dir, run, tags):
    '''
    Sadly, can't use ONNX because this issue ( https://pypi.org/project/skl2onnx/ )
    is only fixed in a later version of skl2onnx, but the AzureML SDK pins an
    old version of that package.
    '''
    fname = 'safe_driver_prediction.pkl'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    path = join(output_dir, fname)
    print(f'Saving serialized model to {path}.')
    joblib.dump(value=model, filename=path)
    run.upload_file(fname, path)
    run.register_model(model_name='safe_driver_prediction', model_path=fname, tags=tags)


def main(args, run):
    data_df = load_input(args.input_dir)
    train, valid = preprocess_split(data_df, args.validation_size, args.random_seed)
    m = train_model(train, valid, args, run)
    tags = evaluate_model(m, valid, run)
    save_model(m, args.output_dir, run, tags)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--learning_rate', default=0.02, type=float)
    parser.add_argument('--min_data', default=100, type=int)
    parser.add_argument('--min_hessian', default=1, type=int)
    parser.add_argument('--num_leaves', default=60, type=int)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--sub_feature', default=0.7, type=float)
    parser.add_argument('--validation_size', default=0.2, type=float)
    parser.add_argument('--verbose', default=4, type=int)

    args = parser.parse_args()
    run = Run.get_context()

    main(args, run)
