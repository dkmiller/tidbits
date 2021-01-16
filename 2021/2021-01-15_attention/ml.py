import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def _clones(module: nn.Module, number: int) -> nn.ModuleList:
    """
    Produce `number` identical layers.
    """
    clones = [copy.deepcopy(module) for _ in range(number)]
    module = nn.ModuleList(clones)
    return module


class EncoderDecoder(nn.Module):
    """
    Standard encoder-decoder architecture.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        source_embedding,
        target_embedding,
        generator,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        """
        Process masked source and target sequences.
        """
        encoded = self.encode(source, source_mask)
        decoded = self.decode(encoded, source_mask, target, target_mask)
        return decoded

    def encode(self, source, source_mask):
        embedded = self.source_embedding(source)
        encoded = self.encoder(embedded, source_mask)
        return encoded

    def decode(self, memory, source_mask, target, target_mask):
        embedded = self.target_embedding(target)
        decoded = self.decoder(embedded, memory, source_mask, target_mask)
        return decoded


class LayerNorm(nn.Module):
    """
    See: https://arxiv.org/abs/1607.06450 for details.
    """

    def __init__(self, num_features: int, ε: float = 1e-6):
        super().__init__()
        ones = torch.ones(num_features)
        zeros = torch.zeros(num_features)
        self.a_2 = nn.Parameter(ones)
        self.b_2 = nn.Parameter(zeros)
        self.ε = ε

    def forward(self, x: torch.Tensor):
        μ = x.mean(-1, keepdim=True)
        σ = x.std(-1, keepdim=True)
        return self.a_2 * (x - μ) / (σ + self.ε) + self.b_2


class Encoder(nn.Module):
    """
    The core encoder is a stack of N layers.
    """

    def __init__(self, layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = _clones(layer, num_layers)
        self.norm = LayerNorm(len(layer))

    def forward(self, x, mask):
        """
        Pass the input and mask through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, mask)

        normalized = self.norm(x)
        return normalized


class Generator(nn.Module):
    """
    Standard linear + softmax generation step.
    """

    def __init__(self, model_dimension: int, vocabulary: int):
        super().__init__()
        self.projection = nn.Linear(model_dimension, vocabulary)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        log = F.log_softmax(projected, dim=-1)
        return log


class SublayerConnection(nn.Module):
    pass


class EncoderLayer(nn.Module):
    pass


class Decoder(nn.Module):
    pass


class DecoderLayer(nn.Module):
    pass


def attention(query, key, value, mask=None, dropout=None):
    pass


class MultiheadedAttention(nn.Module):
    pass


def create_model(source_vocabulary, target_vocabulary, n, d_model, d_ff, h, dropout):
    pass
