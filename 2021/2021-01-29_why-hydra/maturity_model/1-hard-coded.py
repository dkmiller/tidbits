import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


from ml import AutoEncoder


pl.seed_everything(42)

model = AutoEncoder(
    image_size=28,
    color_range=64,
    hyperparam_1=3,
    learning_rate=0.001
)

dataset = MNIST(
    transform=ToTensor(),
    download=True,
    root="."
)

train_loader = DataLoader(
    dataset,
    batch_size=32
)

trainer = pl.Trainer(
    gpus=0,
    log_every_n_steps=200,
    log_gpu_memory="all",
    max_epochs=10,
    num_nodes=11
)

trainer.fit(model, train_loader)
