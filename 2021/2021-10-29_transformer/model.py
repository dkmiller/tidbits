from copy import deepcopy
from dataclasses import dataclass
import math
import numpy as np
from torch import from_numpy, matmul, nn, Tensor
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.sparse import Embedding

from config import Config


@dataclass
class EncoderDecoder(nn.Module):
    encoder: nn.Module
    decoder: nn.Module
    source_embed: nn.Module
    target_embed: nn.Module
    generator: nn.Module

    def encode(self, source, source_mask):
        embedded = self.source_embed(source)
        rv = self.encoder(embedded, source_mask)
        return rv

    def decode(self, memory, source_mask, target, target_mask):
        embedded = self.target_embed(target)
        rv = self.decoder(embedded, memory, source_mask, target_mask)
        return rv

    def forward(self, source, target, source_mask, target_mask):
        encoded = self.encode(source, source_mask)
        rv = self.decode(encoded, source_mask, target, target_mask)
        return rv


class Generator(nn.Module):
    def __init__(self, model_dimension: int, vocab: int):
        self.proj = nn.Linear(model_dimension, vocab)

    def forward(self, x):
        p = self.proj(x)
        rv = F.log_softmax(p, dim=-1)
        return rv


def clones(module: nn.Module, n: int):
    copies = [deepcopy(module) for _ in range(n)]
    rv = nn.ModuleList(copies)
    return rv


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout: float):
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        """
        sublayer = sublayer(self.norm(x))
        rv = x + self.dropout(sublayer)
        return rv


class EncoderLayer(nn.Module):
    def __init__(self, size: int, self_attention, feed_forward, dropout):
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size


class Encoder(nn.Module):
    """
    The core encoder is a stack of `N` layers.
    """

    def __init__(self, layer: EncoderLayer, n: int):
        self.layers = clones(layer, n)
        # Warning: this is using an "official" implementation.
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        Pass the input and mask through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, mask)
        rv = self.norm(x)
        return rv


class DecoderLayer(nn.Module):
    def __init__(
        self, size: int, self_attention, source_attention, feed_forward, dropout
    ):
        self.size = size
        self.self_attention = self_attention
        self.source_attention = source_attention
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        x = self.sublayer[0](x, lambda x: self.self_attention(x, x, x, target_mask))
        x = self.sublayer[1](
            x, lambda x: self.source_attention(x, memory, memory, source_mask)
        )
        rv = self.sublayer[2](x, self.feed_forward)
        return rv


class Decoder(nn.Module):
    """
    Generic `N` layer decoder with masking.
    """

    def __init__(self, layer: DecoderLayer, n: int):
        self.layers = clones(layer, n)
        # Warning: this is using an "official" implementation.
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        rv = self.norm(x)
        return rv


def subsequent_mask(size: int):
    """
    Mask out subsequent positions.
    """
    attention_shape = (1, size, size)
    ones = np.ones(attention_shape)
    subsequent_mask = np.triu(ones, k=1).astype("uint8")
    rv = from_numpy(subsequent_mask) == 0
    return rv


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute scaled dot product attention.
    """
    d_k = query.size(-1)
    scores = matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attention = F.softmax(scores, dim=-1)

    if dropout:
        p_attention = dropout(p_attention)

    rv = matmul(p_attention, value), p_attention

    return rv


class PositionwiseFeedForward(nn.Module):
    """
    Implements the feed-forward-network equation.
    """
    def __init__(self, model_dimension: int, feedforward_dimension: int, dropout: float=0.1):
        self.w_1 = nn.Linear(model_dimension, feedforward_dimension)
        self.w_2 = nn.Linear(feedforward_dimension, model_dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        rv = self.w_2(self.dropout(F.relu(self.w_1(x))))
        return rv


class Model(EncoderDecoder):
    def __init__(self, source_vocab, target_vocab, cfg: Config):
        attention = nn.MultiheadAttention(cfg.h, cfg.model_dimension)
        feedforward = PositionwiseFeedForward(cfg.model_dimension, cfg.d_ff, cfg.dropout)
        position = nn.PositionalEncoding()

        # TODO: finish this...

        encoder_layer = EncoderLayer(cfg.model_dimension, deepcopy(attention), deepcopy(feedforward), cfg.dropout)
        decoder_layer = DecoderLayer(cfg.model_dimension, deepcopy(attention), deepcopy(attention), deepcopy(feedforward), cfg.dropout)
        decoder = Decoder(decoder_layer, cfg.N)
        encoder = Encoder(encoder_layer, cfg.N)

        source_embed = nn.Sequential(Embedding(cfg.model_dimension, source_vocab), deepcopy(positi))
    target_embed: nn.Module
    generator: nn.Module

        super().__init__(encoder, )