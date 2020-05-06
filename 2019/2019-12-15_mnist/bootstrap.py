'''
Download ('bootstrap') initial data files.
'''

import argparse
import logging
import os
import pandas as pd

def download_if_not_exist(pwd, path, url):
    path = os.path.join(pwd, path)
    if os.path.isfile(path):
        logging.info(f'The file {path} is already bootstrapped.')
    else:
        logging.info(f'Downloading {path} from {url}.')
        df = pd.read_csv(url)
        df.to_csv(path)

def main(args):
    # Learned this from my wife. Set the parameter 'format' to configure the
    # appearance of the logs.
    logging.basicConfig(level=logging.DEBUG)

    downloads = {
        'mnist.csv': 'https://www.openml.org/data/get_csv/52667/mnist_784.arff'
    }

    logging.info(f'Boostrapping {len(downloads)} files.')
    for path, url in downloads.items():
        download_if_not_exist(args.directory, path, url)

if __name__ == '__main__':
    # https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser(description='Bootstrap initial data.')
    parser.add_argument('-d', '--directory', dest='directory', default=os.getcwd())
    args = parser.parse_args()

    main(args)

