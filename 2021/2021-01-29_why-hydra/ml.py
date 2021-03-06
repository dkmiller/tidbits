
from typing import Tuple
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch import Tensor


class AutoEncoder(pl.LightningModule):
    def __init__(self, image_size: int, color_range: int, learning_rate: float, hyperparam_1: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(image_size ** 2, color_range),
            nn.ReLU(),
            nn.Linear(color_range, hyperparam_1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hyperparam_1, color_range),
            nn.ReLU(),
            nn.Linear(color_range, image_size ** 2),
        )

        self.example_input_array = None
        self.lr = learning_rate

    def forward(self, x) -> Tensor:
        y = self(x)
        return y

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_id: int) -> Tensor:
        x, y = batch

        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x̂ = self.decoder(z)
        loss = F.mse_loss(x̂, x)
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), self.lr)
        return optimizer
