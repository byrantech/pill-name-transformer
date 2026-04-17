from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _array_to_c(name: str, arr: np.ndarray) -> str:
    flat = arr.reshape(-1)
    values = ", ".join(f"{v:.8f}f" for v in flat[: min(len(flat), 2048)])
    truncated = " /* truncated */" if len(flat) > 2048 else ""
    return (
        f"// shape={list(arr.shape)}\n"
        f"static const float {name}[] = {{ {values} }};{truncated}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("artifacts/model_weights.npz"))
    parser.add_argument("--vocab", type=Path, default=Path("artifacts/vocab.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/model_export.h"))
    args = parser.parse_args()

    bundle = np.load(args.weights)
    vocab = json.loads(args.vocab.read_text(encoding="utf-8"))

    lines = [
        "#pragma once",
        "",
        f"static const int VOCAB_SIZE = {len(vocab)};",
        f"static const char* VOCAB = \"{''.join(vocab)}\";",
        "",
        "// Weights exported from PyTorch for ESP32-side loading or direct arrays.",
        "// For flash size, convert to int8 or int16 in your firmware pipeline.",
        "",
    ]
    for key in sorted(bundle.files):
        c_name = key.replace(".", "_").replace("-", "_")
        lines.append(_array_to_c(c_name, bundle[key]))
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
