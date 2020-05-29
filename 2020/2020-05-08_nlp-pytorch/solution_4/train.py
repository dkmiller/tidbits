import argparse
import logging
from models import MultilayerPerceptron
import os
import torch
from typeguard import typechecked
import yaml

@typechecked
def main(config, log: logging.Logger) -> None:
    hyperparameters = config['hyperparameters']
    torch.manual_seed(hyperparameters['seed'])

    batch_size = 2 # number of samples input at once
    input_dim = 3
    hidden_dim = 100
    output_dim = 4

    model = MultilayerPerceptron([input_dim, hidden_dim, output_dim])
    print(model)

    x_input = torch.rand(batch_size, input_dim)
    print(x_input)

    y = model(x_input, apply_softmax = False)
    print(y)

    y_sm = model(x_input, apply_softmax = True)
    print(y_sm)


if __name__ == '__main__':
    # https://stackoverflow.com/a/5137509
    dir_path = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(dir_path, 'config.yml'))
    parser.add_argument('--level', default='INFO')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logging.basicConfig(level=args.level)
    logger = logging.getLogger()

    main(config, logger)
