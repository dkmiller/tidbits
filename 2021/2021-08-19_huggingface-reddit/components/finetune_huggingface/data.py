import json
from pathlib import Path
from torch.utils.data import Dataset


class Reddit(Dataset):
    def __init__(self, root: str):
        self.files = list(Path(root).rglob("*.json"))

        rows = []

        for f in self.files:
            with f.open("r") as fp:
                lines = fp.readlines()
                rows.extend(map(json.loads, lines))

        for r in rows:
            # print(r["selftext"])
            print(r["gilded"])


reddit = Reddit("/home/azureuser/tmp/")
# print(reddit.files)


from datasets import load_dataset

# https://huggingface.co/docs/datasets/v1.1.3/loading_datasets.html
dataset = load_dataset("json", data_files="/home/azureuser/tmp2/data.json")


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


encoded_dataset = dataset.map(
    lambda examples: tokenizer(
        examples["title_and_text"], truncation=True, padding="max_length"
    ),
    batched=True,
)

encoded_dataset.set_format(type="torch", columns=["input_ids", "n_gilds"])
import torch.utils.data

dataloader = torch.utils.data.DataLoader(encoded_dataset["train"], batch_size=32)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

n_labels = max(dataset["train"]["n_gilds"])

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=n_labels
)


from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# hitting:
# https://discuss.huggingface.co/t/why-am-i-getting-keyerror-loss/6948
