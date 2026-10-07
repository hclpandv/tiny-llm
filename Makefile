.PHONY: help install tokenize inspect dataset model train generate demo clean

help:
	@echo ""
	@echo "Tiny LLM - educational commands"
	@echo ""
	@echo "  make install    Install dependencies"
	@echo "  make tokenize   Train the BPE tokenizer"
	@echo "  make inspect    Inspect tokenizer output"
	@echo "  make dataset    Inspect training batches"
	@echo "  make model      Run model forward/loss demo"
	@echo "  make train      Train the tiny LLM on Alice"
	@echo "  make generate   Generate text"
	@echo "  make demo       Train + generate"
	@echo "  make clean      Remove generated model/tokenizer files"
	@echo ""

install:
	uv sync

tokenize:
	uv run python src/tiny_llm/tokenizer.py

inspect:
	uv run python inspect_tokenizer.py

dataset:
	uv run python src/tiny_llm/dataset.py

model:
	uv run python src/tiny_llm/model.py

train:
	uv run python src/tiny_llm/train.py

generate:
	uv run python src/tiny_llm/generate.py --prompt "queen" --temperature 0.8

stream:
	uv run python src/tiny_llm/generate.py --stream --prompt "Alice was buying"

demo:
	@echo "=== Training tokenizer ==="
	uv run python src/tiny_llm/tokenizer.py
	@echo ""
	@echo "=== Inspecting tokenizer ==="
	uv run python inspect_tokenizer.py
	@echo ""
	@echo "=== Training model ==="
	uv run python src/tiny_llm/train.py
	@echo ""
	@echo "=== Generating text ==="
	uv run python src/tiny_llm/generate.py

clean:
	rm -f data/alice_bpe.json
	rm -f data/tiny_llm.pt