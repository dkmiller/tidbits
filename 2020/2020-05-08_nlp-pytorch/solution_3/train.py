import argparse
from data import raw_reviews, VectorizedReviews
from model import ReviewClassifier
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
import yaml
import pandas as pd


def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device='cpu'):
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict


def get_device(config) -> bool:
    use_gpu = bool(config['compute']['gpu'])
    return torch.device('cuda' if use_gpu else 'cpu')


def load_raw_dataset(config) -> pd.DataFrame:
    file_info = config['files']
    path = file_info['review_json']
    num_reviews = int(file_info['sample_size'])
    print(f'Loading {num_reviews} reviews from {path}.')
    return raw_reviews(path, num_reviews)


def train(config, dataset, device, model: nn.Module, loss, optimizer: Optimizer):
    print('Beginning training...')

    for epoch_index in range(int(config['hyperparameters']['num_epochs'])):
        print(epoch_index)

        batch_generator = generate_batches(dataset, 
                                       batch_size=int(config['hyperparameters']['batch_size']), 
                                       device=device)

        running_loss = 0.0
        running_acc = 0.0
        model.train()
    
        for batch_index, batch_dict in enumerate(batch_generator):
            optimizer.zero_grad()

            ŷ = model(x_in=batch_dict['x_data'].float())
            l = loss(ŷ, batch_dict['y_target'].float())
            loss_batch = l.item()
            print(f'loss = {loss_batch}')

            loss.backward()
            optimizer.step()


def main(config):
    print(f'Configuration: {config}')

    raw_dataset = load_raw_dataset(config)
    print(raw_dataset)

    vectorized_reviews = VectorizedReviews(raw_dataset)
    print(vectorized_reviews)
    num_features = len(vectorized_reviews.vectorizer.vocabulary_)
    print(num_features)

    device = get_device(config)
    print(device)

    model = ReviewClassifier(num_features)
    model = model.to(device)
    print(model)

    loss = nn.BCEWithLogitsLoss()
    print(f'loss = {loss}')

    learning_rate = float(config['hyperparameters']['learning_rate'])
    optimizer = Adam(model.parameters(), lr=learning_rate)
    print(f'optimizer = {optimizer}')

    train(config, vectorized_reviews, device, model, loss, optimizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.BaseLoader)

    main(config)
