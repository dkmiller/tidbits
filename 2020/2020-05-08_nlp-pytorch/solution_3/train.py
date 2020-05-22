import argparse
import yaml
from data import RawReviews


def load_raw_dataset(config) -> RawReviews:
    file_info = config['files']
    path = file_info['review_json']
    num_reviews = int(file_info['sample_size'])
    print(f'Loading {num_reviews} reviews from {path}.')
    return RawReviews(path, num_reviews)


def main(config):
    print(f'Configuration: {config}')

    raw_dataset = load_raw_dataset(config)

    

    print(raw_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.BaseLoader)

    main(config)
