# Tiny LLM

A tiny language model built from scratch for educational purposes.

The goal of this project is not to build a useful production LLM.

The goal is to understand what actually happens inside a language model by
building a small one ourselves.

We use real libraries such as PyTorch and Hugging Face Tokenizers rather than
reimplementing every low-level operation.

The model is trained from scratch on *Alice's Adventures in Wonderland*.

---

## What we build

The project implements the basic lifecycle of an autoregressive language model:

```text
Alice's Adventures in Wonderland
              |
              v
        BPE Tokenizer
              |
              v
           Tokens
              |
              v
       Training Dataset
              |
              v
      Tiny Transformer
              |
              v
      Next-token prediction
              |
              v
       Cross-entropy loss
              |
              v
        Backpropagation
              |
              v
           AdamW
              |
              v
        Trained weights
              |
              v
       Text generation
```
