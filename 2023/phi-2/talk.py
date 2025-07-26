import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


def main():
    if torch.backends.mps.is_available():
        torch.set_default_device("mps")
    else:
        log.warning("M1 accelerator not available, using CPU instead.")

    log.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", torch_dtype="auto", trust_remote_code=True
    )
    log.info("Model is on %s", model.device)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

    while True:
        user_input = input("Say something: ")
        log.info("Tokenizing input...")
        inputs = tokenizer(user_input, return_tensors="pt", return_attention_mask=False)
        log.info("Forward pass...")
        outputs = model.generate(**inputs, max_length=200)
        text = tokenizer.batch_decode(outputs)[0]
        print(text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
