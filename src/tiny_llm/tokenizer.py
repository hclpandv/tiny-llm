from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


DATA = Path("data/alice_clean.txt")
OUTPUT = Path("data/alice_bpe.json")


def main():
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=500,
        min_frequency=2,
        special_tokens=["<unk>"],
    )

    tokenizer.train([str(DATA)], trainer)

    tokenizer.save(str(OUTPUT))

    print(f"Saved tokenizer to {OUTPUT}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()