import argparse
import logging
from transformers import GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel


log = logging.getLogger(__name__)


def main(args):
    text = args.text

    log.info(f"Tokenizing '{text}'")

    tokenizer = GPT2Tokenizer.from_pretrained(args.model)

    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens.input_ids
    log.info(f"Done, found {len(input_ids[0])} tokens")

    model = GPT2LMHeadModel.from_pretrained(args.model)
    outputs = model.generate(
        input_ids=input_ids,
        max_length=args.max_length,
        # This can only be false if num_return_sequences == 1.
        do_sample=True,
        num_return_sequences=args.num_response,
    )

    for i, output in enumerate(outputs):
        example = tokenizer.decode(output, skip_special_tokens=True)
        example = example.replace(args.text, "")
        log.info(f"Example {i} ---- \n\n\t\t{example}\n")


if __name__ == "__main__":
    logging.basicConfig(level="INFO")

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", default=50)
    parser.add_argument(
        "--model", default="gpt2", help="Can also be: gpt2-(medium,large,xl)"
    )
    parser.add_argument("--num_response", default=3)
    parser.add_argument("--text", default="The art of chess has been described as")

    args = parser.parse_args()
    main(args)
