from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer


DATA = Path("data/alice_clean.txt")
TOKENIZER = Path("data/alice_bpe.json")

CONTEXT_LENGTH = 16
BATCH_SIZE = 8


class AliceDataset(Dataset):
    def __init__(self):
        tokenizer = Tokenizer.from_file(str(TOKENIZER))

        text = DATA.read_text(encoding="utf-8")
        encoding = tokenizer.encode(text)

        self.tokens = torch.tensor(
            encoding.ids,
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.tokens) - CONTEXT_LENGTH

    def __getitem__(self, index):
        x = self.tokens[index : index + CONTEXT_LENGTH]
        y = self.tokens[index + 1 : index + CONTEXT_LENGTH + 1]

        return x, y


def main():
    dataset = AliceDataset()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    x, y = next(iter(loader))

    print(f"Number of training examples: {len(dataset):,}")

    print("\nBatch:")
    print("x:")
    print(x)

    print("\ny:")
    print(y)

    print("\nShapes:")
    print("x:", x.shape)
    print("y:", y.shape)


if __name__ == "__main__":
    main()