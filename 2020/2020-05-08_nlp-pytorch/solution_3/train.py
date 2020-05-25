import argparse
from data import raw_reviews, VectorizedReviews
import logging
from model import ReviewClassifier
import torch
import torch.nn as nn
import torch.onnx
from torch.optim import Adam, Optimizer
import yaml
import pandas as pd
from typeguard import typechecked
from typing import Iterable


@typechecked
def generate_batches(dataset, batch_size, device='cpu') -> Iterable[dict]:
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True)
    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = tensor.to(device)
        yield out_data_dict


@typechecked
def get_device(config) -> torch.device:
    use_gpu = bool(config['compute']['gpu'])
    return torch.device('cuda' if use_gpu else 'cpu')


@typechecked
def load_raw_dataset(config, log: logging.Logger) -> pd.DataFrame:
    file_info = config['files']
    path = file_info['review_json']
    num_reviews = int(file_info['sample_size'])

    log.info(f'Loading {num_reviews} reviews from {path}.')
    return raw_reviews(path, num_reviews)


@typechecked
def train(config, dataset, device, model: nn.Module, loss, optimizer: Optimizer, log: logging.Logger) -> None:
    log.info('Beginning training...')

    num_epochs = int(config['hyperparameters']['num_epochs'])
    for epoch_index in range(num_epochs):
        log.info(f'Epoch {epoch_index} / {num_epochs}')

        batch_generator = generate_batches(dataset, 
                                       batch_size=int(config['hyperparameters']['batch_size']), 
                                       device=device)

        running_loss = 0.0
        # running_acc = 0.0
        model.train()
    
        for batch_index, batch_dict in enumerate(batch_generator):
            optimizer.zero_grad()


            if batch_index == 0:
                log.debug('ONNX export')
                torch.onnx.export(model, batch_dict['x_data'].float(), "model.onnx", export_params=True, opset_version=10, do_constant_folding=True)

            x = batch_dict['x_data'].float()
            y = batch_dict['y_target'].float()
            ŷ = model(x)
            l = loss(ŷ, y)

            loss_batch = l.item()
            log.debug(f'loss batch = {loss_batch}')

            running_loss += (loss_batch - running_loss) / (batch_index + 1)
            if batch_index % 20 == 0:
                log.info(f'index {batch_index}, running loss = {running_loss}')
    
            l.backward()
            optimizer.step()
    

@typechecked
def main(config, log: logging.Logger) -> None:
    log.debug(f'Configuration: {config}')

    raw_dataset = load_raw_dataset(config, log)
    log.debug(f'size of raw dataset = {len(raw_dataset)}')

    max_features = int(config['hyperparameters']['max_features'])
    log.debug(f'Limiting to {max_features} features.')

    vectorized_reviews = VectorizedReviews(raw_dataset, max_features)
    log.debug(f'size of vectorized reviews = {len(vectorized_reviews)}')

    num_features = len(vectorized_reviews.vectorizer.vocabulary_)
    log.debug(f'Num features = {num_features}')

    device = get_device(config)
    log.info(f'Using device = {device}')

    model = ReviewClassifier(num_features)
    model = model.to(device)
    log.info(model)

    loss = nn.BCEWithLogitsLoss()
    log.info(f'loss = {loss}')

    learning_rate = float(config['hyperparameters']['learning_rate'])
    optimizer = Adam(model.parameters(), lr=learning_rate)
    log.info(f'optimizer = {optimizer}')

    train(config, vectorized_reviews, device, model, loss, optimizer, log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml')
    parser.add_argument('--level', default='INFO')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.BaseLoader)

    logging.basicConfig(level=args.level)
    logger = logging.getLogger()

    main(config, logger)
