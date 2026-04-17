from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.model import TinyNameTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.pt"))
    parser.add_argument("--vocab", type=Path, default=Path("artifacts/vocab.json"))
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=23)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args()

    vocab = json.loads(args.vocab.read_text(encoding="utf-8"))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    model = TinyNameTransformer(
        vocab_size=len(vocab),
        d_model=args.d_model,
        n_head=args.heads,
        n_layer=args.layers,
        max_len=args.max_len,
    )
    state = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state)

    print("Generated fake prescription-style names:")
    seen = set()
    attempts = 0
    while len(seen) < args.count and attempts < args.count * 20:
        candidate = model.sample(
            stoi=stoi,
            itos=vocab,
            prompt=args.prompt,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        attempts += 1
        if 4 <= len(candidate) <= 16 and candidate.isascii():
            seen.add(candidate)
    for i, name in enumerate(sorted(seen), start=1):
        print(f"{i:02d}. {name}")


if __name__ == "__main__":
    main()
