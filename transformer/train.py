from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data import fetch_openfda_names, save_names
from src.model import (
    ArtifactPaths,
    SPECIAL_PAD,
    TinyNameTransformer,
    build_dataset,
    build_vocab,
    save_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/drug_names.json"))
    parser.add_argument("--max-records", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Fetching real drug names from openFDA...")
    names = fetch_openfda_names(max_records=args.max_records)
    if not names:
        raise RuntimeError("No names fetched from openFDA.")
    save_names(names, args.data)
    print(f"Fetched and saved {len(names)} names to {args.data}")

    vocab = build_vocab(names)
    stoi = {ch: i for i, ch in enumerate(vocab)}
    x, y = build_dataset(names, stoi, args.max_len)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = TinyNameTransformer(
        vocab_size=len(vocab),
        d_model=args.d_model,
        n_head=args.heads,
        n_layer=args.layers,
        max_len=args.max_len - 1,
    ).to(device)

    x = x.to(device)
    y = y.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    pad_id = stoi[SPECIAL_PAD]

    print(f"Training on device={device} with {x.size(0)} samples...")
    for epoch in range(args.epochs):
        perm = torch.randperm(x.size(0), device=device)
        x_shuf, y_shuf = x[perm], y[perm]
        running = 0.0
        steps = 0
        for i in range(0, x_shuf.size(0), args.batch_size):
            xb = x_shuf[i : i + args.batch_size]
            yb = y_shuf[i : i + args.batch_size]
            logits = model(xb)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                yb.reshape(-1),
                ignore_index=pad_id,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()
            steps += 1
        print(f"Epoch {epoch + 1:02d}/{args.epochs}: loss={running / max(steps, 1):.4f}")

    out = ArtifactPaths(
        model_pt=Path("artifacts/model.pt"),
        vocab_json=Path("artifacts/vocab.json"),
        export_npz=Path("artifacts/model_weights.npz"),
    )
    save_artifacts(model, vocab, out)
    print("Saved artifacts:")
    print(f" - {out.model_pt}")
    print(f" - {out.vocab_json}")
    print(f" - {out.export_npz}")


if __name__ == "__main__":
    main()
