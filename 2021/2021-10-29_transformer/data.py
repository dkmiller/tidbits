import spacy
from torchtext.datasets import IWSLT2016
from typing import Tuple

from config import Config


def tokenizer(package: str):
    model = spacy.load(package)

    def rv(text):
        return [tok.text for tok in model.tokenizer(text)]

    return rv


def source_and_target(cfg: Config):
    # TODO: should that come from config?
    en = tokenizer(cfg.english)
    de = tokenizer(cfg.german)

    # TODO: why are these (should they be) different?
    # source = data.Field(tokenize=en, pad_token=cfg.BLANK_WORD)
    # target = data.Field(tokenize=de, pad_token=cfg.BLANK_WORD, init_token=cfg.BOS_WORD, eos_token=cfg.EOS_WORD)

    # train, _, _ = datasets.IWSLT.splits()
