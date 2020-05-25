import argparse
from data import raw_reviews, VectorizedReviews
import logging
from model import ReviewClassifier
import torch
import torch.nn as nn
import torch.onnx
from torch.optim import Adam, Optimizer
from torch.utils.tensorboard import SummaryWriter
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
def log_compute_utilization(device: torch.device, index: int, writer: SummaryWriter) -> None:
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    memory_usage = float(allocated_memory) / total_memory

    writer.add_scalar(f'compute/{device}/usage', memory_usage, index)


@typechecked
def train(config, dataset, device, model: nn.Module, loss, optimizer: Optimizer, log: logging.Logger, writer: SummaryWriter) -> None:
    log.info('Beginning training...')
    hyperparameters = config['hyperparameters']
    counter = 0

    num_epochs = hyperparameters['num_epochs']
    for epoch_index in range(num_epochs):
        log.info(f'Epoch {epoch_index} / {num_epochs}')

        batch_generator = generate_batches(dataset, 
                                       batch_size=hyperparameters['batch_size'], 
                                       device=device)

        running_loss = 0.0
        model.train()
    
        batch_size = len(dataset) / num_epochs
        for batch_index, batch_dict in enumerate(batch_generator):
            optimizer.zero_grad()

            x = batch_dict['x_data'].float()
            y = batch_dict['y_target'].float()

            if batch_index == 0:
                file = config['files']['model_state_file']
                log.debug(f'ONNX export to {file}')
                torch.onnx.export(model, x, file, export_params=True)

            ŷ = model(x)
            l = loss(ŷ, y)

            loss_batch = l.item()
            log.debug(f'loss batch = {loss_batch}')

            running_loss += (loss_batch - running_loss) / (batch_index + 1)
            writer.add_scalar('loss/train/running', running_loss, counter)
            writer.add_scalar('loss/train/batch', loss_batch, counter)

            log_compute_utilization(device, counter, writer)

            if batch_index % 20 == 0:
                log.info(f'index {batch_index} / {batch_size}, running loss = {running_loss}')
    
            l.backward()
            optimizer.step()

            counter += 1
    

@typechecked
def main(config, log: logging.Logger) -> None:
    log.debug(f'Configuration: {config}')
    hyperparameters = config['hyperparameters']

    raw_dataset = load_raw_dataset(config, log)
    log.info(f'size of raw dataset = {len(raw_dataset)}')

    max_features = hyperparameters['max_features']
    log.info(f'Limiting to {max_features} features.')

    vectorized_reviews = VectorizedReviews(raw_dataset, max_features)
    log.info(f'size of vectorized reviews = {len(vectorized_reviews)}')

    num_text_features = len(vectorized_reviews.text_vectorizer.vocabulary_)
    log.info(f'Num review tokens = {num_text_features}')

    num_rating_features = len(vectorized_reviews.rating_vectorizer.vocabulary_)
    log.info(f'Rating vocabulary size = {num_rating_features}')

    device = get_device(config)
    log.info(f'Using device = {device}')

    model = ReviewClassifier(num_text_features)
    model = model.to(device)
    log.info(model)

    loss = nn.BCEWithLogitsLoss()
    log.info(f'loss = {loss}')

    learning_rate = hyperparameters['learning_rate']
    optimizer = Adam(model.parameters(), lr=learning_rate)
    log.info(f'optimizer = {optimizer}')

    tensorboard_directory = config['files']['log_dir']
    log.info(f'Writing Tensorboard logs to {tensorboard_directory}.')

    seed = hyperparameters['seed']
    torch.manual_seed(seed)
    log.info(f'Using seed {seed}.')

    writer = SummaryWriter(log_dir=tensorboard_directory)

    train(config, vectorized_reviews, device, model, loss, optimizer, log, writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml')
    parser.add_argument('--level', default='INFO')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logging.basicConfig(level=args.level)
    logger = logging.getLogger()

    main(config, logger)
