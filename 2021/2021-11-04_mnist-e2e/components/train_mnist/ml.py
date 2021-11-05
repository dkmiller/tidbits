import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam


class MnistEncoder(pl.LightningModule):
    """
    https://pytorch-lightning.readthedocs.io/en/latest/starter/new-project.html

    https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(self, image_dim: int, hp1: int, hp2: int, lr: float = 1e-3):
        super().__init__()
        image_size = image_dim * image_dim

        self.encoder = nn.Sequential(
            nn.Linear(image_size, hp1), nn.ReLU(), nn.Linear(hp1, hp2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hp2, hp1), nn.ReLU(), nn.Linear(hp1, image_size)
        )
        self.learning_rate = lr

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x̂ = self.decoder(z)
        loss = F.mse_loss(x̂, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
