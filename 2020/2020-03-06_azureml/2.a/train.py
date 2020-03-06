import argparse
import glob
import joblib
import lightgbm
import numpy as np
import pandas as pd
from azureml.core.run import Run
from os.path import join
from sklearn.model_selection import train_test_split
from sklearn import metrics


def load_input(path):
    '''
    Follow https://pymotw.com/2/glob/ to use Glob for finding Csv file.
    '''
    print(f'Searching {args.input_dir} for CSV files.')
    file = glob.glob(join(args.input_dir, '*.csv'), recursive=True)[0]
    print(f'Loading data from "{file}".')

    df = pd.read_csv(file)
    print(f'Data shape: {df.shape}')

    return df


def preprocess_split(df, validation_size, seed):
    features = df.drop(['target', 'id'], axis=1)
    labels = np.array(df['target'])
    features_train, features_valid, labels_train, labels_valid = train_test_split(
        features, labels, test_size=validation_size, random_state=seed)

    train_data = lightgbm.Dataset(features_train, label=labels_train)
    valid_data = lightgbm.Dataset(
        features_valid, label=labels_valid, free_raw_data=False)

    return (train_data, valid_data)


def evaluate_model(model, valid, run):
    predictions = model.predict(valid.data)
    fpr, tpr, _ = metrics.roc_curve(valid.label, predictions)
    run.log('auc', metrics.auc(fpr, tpr))


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


def save_model(model, output_dir):
    '''
    Sadly, can't use ONNX because this issue ( https://pypi.org/project/skl2onnx/ )
    is only fixed in a later version of skl2onnx, but the AzureML SDK pins an
    old version of that package.
    '''
    fname = join(output_dir, 'safe_driver_prediction.pkl')
    print(f'Saving serialized model to {fname}.')
    joblib.dump(value=model, filename=fname)


def main(args, run):
    data_df = load_input(args.input_dir)
    train, valid = preprocess_split(
        data_df, args.validation_size, args.random_seed)
    m = train_model(train, valid, args, run)
    evaluate_model(m, valid, run)
    save_model(m, args.output_dir)


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
