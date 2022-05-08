"""
https://huggingface.co/docs/transformers/custom_datasets#finetune-with-the-trainer-api
"""


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)
