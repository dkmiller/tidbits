"""
Data-related manipulation. Imitates:

https://www.kaggle.com/atamazian/birdclef-2022-lightning-model-training
"""


from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class BirdclefData(Dataset):
    """
    https://www.kaggle.com/c/birdclef-2022/data
    """

    def __init__(self, root: str):
        self.root = Path(root)

    def __len__(self):
        pass

    def __getitem__(self):
        pass


class BirdclefLoader(DataLoader):
    pass
