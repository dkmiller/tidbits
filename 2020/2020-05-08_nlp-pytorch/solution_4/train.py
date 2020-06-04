import argparse
import data
import logging
from models import MultilayerPerceptron
import os
import pandas as pd
from pyconfigurableml.entry import run
import torch
from typeguard import typechecked
import yaml


@typechecked
def get_device(hpms) -> torch.device:
    use_gpu = hpms['gpu']
    return torch.device('cuda' if use_gpu else 'cpu')


def main(config, log: logging.Logger) -> None:
    hyperparameters = config['hyperparameters']
    torch.manual_seed(hyperparameters['seed'])

    batch_size = 2 # number of samples input at once
    input_dim = 3
    hidden_dim = hyperparameters['hidden_dim']
    output_dim = 4

    df = pd.read_csv(config['data']['surname_csv'])
    log.info(df)

    dataset = data.Surnames(df)
    log.info(dataset[0])

    device = get_device(hyperparameters)
    log.info(device)

    model = MultilayerPerceptron([input_dim, hidden_dim, output_dim]).to(device)
    log.info(model)

    x_input = torch.rand(batch_size, input_dim)
    log.info(x_input)

    y = model(x_input, apply_softmax = False)
    log.info(y)

    y_sm = model(x_input, apply_softmax = True)
    log.info(y_sm)

    loss = torch.nn.CrossEntropyLoss(dataset.class_weights())
    log.info(loss)

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
    log.info(optimizer)


run(main, __file__, __name__)
