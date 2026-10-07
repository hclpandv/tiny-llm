from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import AliceDataset, BATCH_SIZE
from model import TinyLanguageModel, VOCAB_SIZE


NUM_EPOCHS = 5
LEARNING_RATE = 0.001

CHECKPOINT = Path("data/tiny_llm.pt")


def main():
    # -------------------------
    # Dataset
    # -------------------------

    dataset = AliceDataset()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    print(f"Training examples: {len(dataset):,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Batches per epoch: {len(loader):,}")

    # -------------------------
    # Model
    # -------------------------

    model = TinyLanguageModel()

    print(
        f"Model parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # -------------------------
    # Loss and optimizer
    # -------------------------

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # -------------------------
    # Training
    # -------------------------

    for epoch in range(NUM_EPOCHS):

        model.train()

        total_loss = 0.0

        for step, (x, y) in enumerate(loader):

            # Forward pass
            logits = model(x)

            logits = logits.view(
                -1,
                VOCAB_SIZE,
            )

            y = y.view(-1)

            loss = loss_function(
                logits,
                y,
            )

            # Backward pass
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            if step % 500 == 0:
                print(
                    f"Epoch {epoch + 1}/{NUM_EPOCHS} "
                    f"| Step {step}/{len(loader)} "
                    f"| Loss {loss.item():.4f}"
                )

        average_loss = total_loss / len(loader)

        print(
            f"\nEpoch {epoch + 1} complete "
            f"| Average loss: {average_loss:.4f}\n"
        )

    # -------------------------
    # Save checkpoint
    # -------------------------

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab_size": VOCAB_SIZE,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "final_loss": average_loss,
    }

    torch.save(
        checkpoint,
        CHECKPOINT,
    )

    print(f"Saved checkpoint to {CHECKPOINT}")


if __name__ == "__main__":
    main()