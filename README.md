<div align="center">

# 🧠 Tiny LLM

### Build a language model from scratch, and finally understand what's inside.
**BPE tokenizer → Transformer → Training → Text generation**
All trained on *Alice's Adventures in Wonderland* 🐇

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Transformer-EE4C2C?logo=pytorch&logoColor=white)
![Tokenizers](https://img.shields.io/badge/🤗_Tokenizers-BPE-FFD21E)
![Purpose](https://img.shields.io/badge/Purpose-Educational-8A2BE2)
![uv](https://img.shields.io/badge/Built_with-uv-DE5FE9)

</div>

---

## 🎯 The Goal

This project is **not** about building a useful production LLM.

It is about understanding what actually happens inside a language model by **building a small one yourself**, with real libraries (PyTorch and Hugging Face Tokenizers) rather than reimplementing every low-level operation.

> 📖 The model is trained from scratch on *Alice's Adventures in Wonderland*, small enough to train on a laptop and fun enough to read the output.

---

## 🔁 The Full Lifecycle, End to End

```
Alice's Adventures in Wonderland
              │
              ▼
        BPE Tokenizer
              │
              ▼
           Tokens
              │
              ▼
       Training Dataset
              │
              ▼
      Tiny Transformer
              │
              ▼
    Next-token prediction
              │
              ▼
     Cross-entropy loss
              │
              ▼
       Backpropagation
              │
              ▼
          AdamW
              │
              ▼
       Trained weights
              │
              ▼
       Text generation ✨
```

Every stage has its own script, so you can run it, read it and poke at it on its own.

---

## ⚡ Quick Start

**Requirements:** Python 3.11+ and [`uv`](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/hclpandv/tiny-llm.git
cd tiny-llm
make install
make demo      # train tokenizer → train model → generate text
```

---

## 🧪 Explore Step by Step

| Command | What it does |
|---|---|
| `make install` | 📦 Install dependencies |
| `make tokenize` | 🔤 Train the BPE tokenizer |
| `make inspect` | 🔍 Inspect tokenizer output |
| `make dataset` | 🗂️ Inspect training batches |
| `make model` | 🧮 Run model forward pass and loss demo |
| `make train` | 🏋️ Train the tiny LLM on Alice |
| `make generate` | ✍️ Generate text from a prompt |
| `make stream` | 🌊 Stream generated text token by token |
| `make demo` | 🚀 Train and generate, everything in one go |
| `make clean` | 🧹 Remove generated model and tokenizer files |

### Try your own prompt

```bash
uv run python src/tiny_llm/generate.py --prompt "queen" --temperature 0.8
uv run python src/tiny_llm/generate.py --stream --prompt "Alice was buying"
```

> 🌡️ Lower the `--temperature` for safer text, raise it for wilder, more "Wonderland" output.

---

## 🧭 What You'll Learn

- 🔤 How **BPE tokenization** turns text into token IDs
- 🗂️ How a text corpus becomes **next-token training examples**
- 🧠 How a **Transformer** predicts the next token
- 📉 How **cross-entropy loss, backpropagation and AdamW** train the model
- ✍️ How **temperature** and autoregressive sampling generate text

---

## 🔗 Companion Project

Want to *see* inside a model while it runs? Check out **[LLM-Xray 🔬](https://github.com/hclpandv/llm-xray)**, an interactive visualizer for tokens, embeddings, per-layer tensors and next-token probabilities.

---

## 🤝 Contributing

Ideas, fixes and learning-oriented improvements are welcome. Open an issue or a PR.

## 📄 License

License to be decided.

<div align="center">

**If this helped you understand LLMs, give it a ⭐**

</div>
