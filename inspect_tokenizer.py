from tokenizers import Tokenizer

t = Tokenizer.from_file("data/alice_bpe.json")

for text in [
    "Alice was beginning to get very tired",
    "Wonderland",
    "rabbit",
    "conversations",
]:
    e = t.encode(text)
    print(repr(text))
    print(e.tokens)
    print(e.ids)
    print()
