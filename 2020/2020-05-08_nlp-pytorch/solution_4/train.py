import argparse
import data
import logging
from models import MultilayerPerceptron
import os
import pandas as pd
from pyconfigurableml.entry import run
import torch
from typeguard import typechecked
from typing import Iterable
import yaml


@typechecked
def get_device(hpms) -> torch.device:
    use_gpu = hpms['gpu']
    return torch.device('cuda' if use_gpu else 'cpu')


@typechecked
def generate_batches(dataset, batch_size, device='cpu') -> Iterable[dict]:
    '''
    TODO: use common library between solutions 3 and 4.
    '''
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True)
    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            print(f'{name} -> {tensor.shape}')
            out_data_dict[name] = tensor.to(device)
        yield out_data_dict



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

    # TODO: input and output dimensions should come from the data.
    model = MultilayerPerceptron([
        len(dataset.surname_vectorizer.vocabulary_),
        hidden_dim,
        len(dataset.nationality_vectorizer.vocabulary_)
        ]).to(device)
    log.info(model)

    # x_input = torch.rand(batch_size, len(dataset.surname_vectorizer.vocabulary_)).to(device)
    # log.info(x_input)

    # y = model(x_input, apply_softmax = False)
    # log.info(y)

    # y_sm = model(x_input, apply_softmax = True)
    # log.info(y_sm)

    loss = torch.nn.CrossEntropyLoss(dataset.class_weights().to(device))
    log.info(loss)

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
    log.info(optimizer)

    num_epochs = hyperparameters['num_epochs']
    for epoch_index in range(num_epochs):
        log.info(f'Epoch {epoch_index} / {num_epochs}')

        batch_generator = generate_batches(dataset, 
                                       batch_size=hyperparameters['batch_size'], 
                                       device=device)

        model.train()
        for batch_index, batch_dict in enumerate(batch_generator):
            optimizer.zero_grad()


            x = batch_dict['x_data'].float()
            y = batch_dict['y_target']

            ŷ = model(x)

            log.info(x.shape)
            log.info(ŷ.shape)
            log.info(y.shape)
            
            l = loss(ŷ, y)
            loss_batch = l.to('cpu').item()
            log.info(loss_batch)


    
run(main, __file__, __name__)
