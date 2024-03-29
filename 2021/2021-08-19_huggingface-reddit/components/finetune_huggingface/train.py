"""
https://huggingface.co/docs/transformers/custom_datasets#finetune-with-the-trainer-api
"""


from argparse_dataclass import ArgumentParser
from dataclasses import dataclass
from datasets import load_dataset
import logging
from os.path import join
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
)


log = logging.getLogger(__name__)


@dataclass
class Args:
    input_dir: str
    input_file: str
    model: str = "distilbert-base-uncased"


def coalesce(example):
    """
    TODO: why is this needed?

    https://huggingface.co/docs/datasets/process#map
    """
    example["label"] = int(bool(example["label"]))
    return example


def main(train_args: TrainingArguments, args: Args):
    log.info(f"Parsed args: {args}")
    log.info(f"Parsed training args: {train_args}")

    # https://huggingface.co/docs/datasets/loading#json
    dataset = load_dataset("json", data_files=join(args.input_dir, args.input_file))
    coalesced_dataset = dataset.map(coalesce)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_dataset = coalesced_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    # TODO: separate train and eval inputs.
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    parser = HfArgumentParser(TrainingArguments)
    (train_args, unknown) = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    parser = ArgumentParser(Args)
    args = parser.parse_args(unknown)

    main(train_args, args)
