from dataclasses import dataclass
import numpy as np
from pathlib import Path
import pytorch_lightning as pl
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Optional, Tuple


class MnistDataset(Dataset):
    """
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """

    def __init__(self, image_dir: Path, label_dir: Path):
        # https://stackoverflow.com/a/59274557
        image_path = next(image_dir.glob("*.npy"))
        label_path = next(label_dir.glob("*.npy"))

        self.images: np.ndarray = np.load(image_path)
        self.labels: np.ndarray = np.load(label_path)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index) -> Tuple[Tensor, int]:
        image = Tensor(self.images[index])
        label = self.labels[index]
        return (image, label)


@dataclass
class Mnist(pl.LightningDataModule):
    """
    https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    batch_size: int
    train_val_ratio: float
    train_image_dir: Path
    train_label_dir: Path
    test_image_dir: Path
    test_label_dir: Path

    def setup(self, stage: Optional[str] = None) -> None:
        train_full = MnistDataset(self.train_image_dir, self.train_label_dir)
        train_count = int(self.train_val_ratio * len(train_full))
        test_count = len(train_full) - train_count

        train_data, val_data = random_split(train_full, [train_count, test_count])

        self.train_loader = DataLoader(train_data, self.batch_size)
        self.val_loader = DataLoader(val_data, self.batch_size)

        test_data = MnistDataset(self.test_image_dir, self.test_label_dir)
        self.test_loader = DataLoader(test_data, self.batch_size)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
