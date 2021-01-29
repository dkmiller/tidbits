import argparse
import logging
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


from ml import AutoEncoder


def get_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int)

    model = parser.add_argument_group("model hyperparameters")
    model.add_argument("--image_size", default=28, type=int)
    model.add_argument("--color_range", default=64, type=int)
    model.add_argument("--hyperparam_1", default=3, type=int)
    model.add_argument("--learning_rate", default=0.001, type=float)

    dataset = parser.add_argument_group("dataset configuration")
    dataset.add_argument("--download", default=True, type=bool)
    dataset.add_argument("--root", default=".", type=str)

    dataset = parser.add_argument_group("dataloader configuration")
    dataset.add_argument("--batch_size", default=32, type=int)

    dataset = parser.add_argument_group("trainer configuration")
    dataset.add_argument("--gpus", default=0, type=int)
    dataset.add_argument("--log_every_n_steps", default=200, type=int)
    dataset.add_argument("--log_gpu_memory", default="all", type=str)
    dataset.add_argument("--max_epochs", default=10, type=int)
    dataset.add_argument("--num_nodes", default=1, type=int)

    return parser


def main(args):
    logging.basicConfig(level="INFO")
    logging.info(f"Arguments: {args}")

    pl.seed_everything(args.seed)

    model = AutoEncoder(
        image_size=args.image_size,
        color_range=args.color_range,
        hyperparam_1=args.hyperparam_1,
        learning_rate=args.learning_rate
    )

    dataset = MNIST(
        transform=ToTensor(),
        download=args.download,
        root=args.root
    )

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size
    )

    trainer = pl.Trainer(
        gpus=args.gpus,
        log_every_n_steps=args.log_every_n_steps,
        log_gpu_memory=args.log_gpu_memory,
        max_epochs=args.max_epochs,
        num_nodes=args.num_nodes
    )

    trainer.fit(model, train_loader)


if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    main(args)
