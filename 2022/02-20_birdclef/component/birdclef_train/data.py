"""
Data-related manipulation. Imitates:

https://www.kaggle.com/atamazian/birdclef-2022-lightning-model-training
"""


from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader


# https://github.com/PyTorchLightning/pytorch-lightning/issues/8272
@dataclass
class BirdclefData(Dataset):
    pass


class BirdclefLoader(DataLoader):
    pass
