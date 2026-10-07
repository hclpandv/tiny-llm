import torch
import torch.nn as nn
import torch.nn.functional as F


VOCAB_SIZE = 500
CONTEXT_LENGTH = 16

EMBEDDING_DIM = 128
NUM_HEADS = 4
FFN_DIM = 512

NUM_BLOCKS = 2


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

        assert EMBEDDING_DIM % NUM_HEADS == 0

        self.head_dim = EMBEDDING_DIM // NUM_HEADS

        self.q_projection = nn.Linear(
            EMBEDDING_DIM,
            EMBEDDING_DIM,
        )

        self.k_projection = nn.Linear(
            EMBEDDING_DIM,
            EMBEDDING_DIM,
        )

        self.v_projection = nn.Linear(
            EMBEDDING_DIM,
            EMBEDDING_DIM,
        )

        self.output_projection = nn.Linear(
            EMBEDDING_DIM,
            EMBEDDING_DIM,
        )

        mask = torch.tril(
            torch.ones(
                CONTEXT_LENGTH,
                CONTEXT_LENGTH,
            )
        )

        self.register_buffer(
            "causal_mask",
            mask,
        )

    def forward(self, x):
        batch_size, sequence_length, embedding_dim = x.shape

        q = self.q_projection(x)
        k = self.k_projection(x)
        v = self.v_projection(x)

        q = q.view(
            batch_size,
            sequence_length,
            NUM_HEADS,
            self.head_dim,
        )

        k = k.view(
            batch_size,
            sequence_length,
            NUM_HEADS,
            self.head_dim,
        )

        v = v.view(
            batch_size,
            sequence_length,
            NUM_HEADS,
            self.head_dim,
        )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = q @ k.transpose(-2, -1)

        scores = scores / (self.head_dim ** 0.5)

        mask = self.causal_mask[
            :sequence_length,
            :sequence_length,
        ]

        scores = scores.masked_fill(
            mask == 0,
            float("-inf"),
        )

        attention_weights = F.softmax(
            scores,
            dim=-1,
        )

        output = attention_weights @ v

        output = output.transpose(1, 2)

        output = output.contiguous().view(
            batch_size,
            sequence_length,
            embedding_dim,
        )

        return self.output_projection(output)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                EMBEDDING_DIM,
                FFN_DIM,
            ),
            nn.GELU(),
            nn.Linear(
                FFN_DIM,
                EMBEDDING_DIM,
            ),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(
            EMBEDDING_DIM,
        )

        self.attention = CausalSelfAttention()

        self.layer_norm_2 = nn.LayerNorm(
            EMBEDDING_DIM,
        )

        self.feed_forward = FeedForward()

    def forward(self, x):
        x = x + self.attention(
            self.layer_norm_1(x)
        )

        x = x + self.feed_forward(
            self.layer_norm_2(x)
        )

        return x


class TinyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(
            VOCAB_SIZE,
            EMBEDDING_DIM,
        )

        self.position_embedding = nn.Embedding(
            CONTEXT_LENGTH,
            EMBEDDING_DIM,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock()
                for _ in range(NUM_BLOCKS)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(
            EMBEDDING_DIM,
        )

        self.output_projection = nn.Linear(
            EMBEDDING_DIM,
            VOCAB_SIZE,
        )

    def forward(self, x):
        batch_size, sequence_length = x.shape

        positions = torch.arange(
            sequence_length,
            device=x.device,
        )

        token_vectors = self.token_embedding(x)

        position_vectors = self.position_embedding(
            positions
        )

        x = token_vectors + position_vectors

        for block in self.blocks:
            x = block(x)

        x = self.final_layer_norm(x)

        logits = self.output_projection(x)

        return logits


def main():
    model = TinyLanguageModel()

    x = torch.tensor([
        [13, 18, 11, 26],
        [30, 15, 28, 19],
    ])

    y = torch.tensor([
        [18, 11, 26, 30],
        [15, 28, 19, 7],
    ])

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
    )

    # -------------------------
    # Before training
    # -------------------------

    logits = model(x)

    loss = loss_function(
        logits.view(-1, VOCAB_SIZE),
        y.view(-1),
    )

    print("Loss before training:")
    print(loss.item())

    # -------------------------
    # Backpropagation
    # -------------------------

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    # -------------------------
    # After one update
    # -------------------------

    logits = model(x)

    loss = loss_function(
        logits.view(-1, VOCAB_SIZE),
        y.view(-1),
    )

    print("\nLoss after one update:")
    print(loss.item())

    print("\nNumber of parameters:")
    print(sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    main()