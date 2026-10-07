import argparse
import time

import torch
from tokenizers import Tokenizer

from model import TinyLanguageModel
from model import CONTEXT_LENGTH


CHECKPOINT = "data/tiny_llm.pt"
TOKENIZER_PATH = "data/alice_bpe.json"

DEFAULT_PROMPT = "Alice was"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_NEW_TOKENS = 50


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    temperature,
    stream=False,
):
    encoding = tokenizer.encode(prompt)

    tokens = torch.tensor(
        [encoding.ids],
        dtype=torch.long,
    )

    model.eval()

    if stream:
        print(prompt, end="", flush=True)

    previous_text = prompt

    with torch.no_grad():

        for _ in range(max_new_tokens):

            input_tokens = tokens[:, -CONTEXT_LENGTH:]

            logits = model(input_tokens)

            next_token_logits = logits[:, -1, :]

            next_token_logits = (
                next_token_logits / temperature
            )

            probabilities = torch.softmax(
                next_token_logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            tokens = torch.cat(
                [tokens, next_token],
                dim=1,
            )

            if stream:
                generated_ids = tokens[0].tolist()

                current_text = tokenizer.decode(
                    generated_ids
                )

                new_text = current_text[
                    len(previous_text):
                ]

                print(
                    new_text,
                    end="",
                    flush=True,
                )

                previous_text = current_text

                time.sleep(0.2)

    generated_ids = tokens[0].tolist()

    return tokenizer.decode(generated_ids)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text with the tiny LLM."
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Text to start generation from.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature.",
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Number of new tokens to generate.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation.",
    )

    parser.add_argument(
        "--stream",
        action="store_true",
        help="Show text as it is generated.",
    )

    args = parser.parse_args()

    if args.temperature <= 0:
        parser.error(
            "--temperature must be greater than 0"
        )

    if args.tokens <= 0:
        parser.error(
            "--tokens must be greater than 0"
        )

    return args


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    tokenizer = Tokenizer.from_file(
        TOKENIZER_PATH
    )

    model = TinyLanguageModel()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if args.stream:
        print("\nGenerating:\n")

        generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            stream=True,
        )

        print()

    else:
        text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
        )

        print("\nGenerated text:\n")
        print(text)


if __name__ == "__main__":
    main()