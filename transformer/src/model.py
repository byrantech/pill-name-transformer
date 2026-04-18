from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


SPECIAL_PAD = "_"
SPECIAL_BOS = "^"
SPECIAL_EOS = "$"


def build_vocab(drug_names: list[str]) -> list[str]:
    alphabet = set("".join(drug_names))
    alphabet.add(" ")
    symbols = [SPECIAL_PAD, SPECIAL_BOS, SPECIAL_EOS] + sorted(alphabet)
    return symbols


def encode_name(name: str, stoi: dict[str, int], max_len: int) -> list[int]:
    tokens = [stoi[SPECIAL_BOS]] + [stoi[c] for c in name] + [stoi[SPECIAL_EOS]]
    if len(tokens) < max_len:
        tokens = tokens + [stoi[SPECIAL_PAD]] * (max_len - len(tokens))
    return tokens[:max_len]


def build_dataset(names: list[str], stoi: dict[str, int], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [encode_name(name, stoi, max_len) for name in names]
    x = torch.tensor([row[:-1] for row in encoded], dtype=torch.long)
    y = torch.tensor([row[1:] for row in encoded], dtype=torch.long)
    return x, y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TinyNameTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_head: int = 4, n_layer: int = 2, max_len: int = 32):
        super().__init__()
        self.max_len = max_len
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layer)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.token_embed(x)
        t = self.pos_enc(t)
        causal_mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), diagonal=1).bool()
        t = self.encoder(t, mask=causal_mask)
        t = self.norm(t)
        return self.head(t)

    @torch.no_grad()
    def sample(
        self,
        stoi: dict[str, int],
        itos: list[str],
        prompt: str = "",
        temperature: float = 0.9,
        top_k: int = 8,
    ) -> str:
        self.eval()
        seq = [stoi[SPECIAL_BOS]]
        seq.extend(stoi.get(ch, stoi[" "]) for ch in prompt.lower())
        while len(seq) < self.max_len:
            x = torch.tensor([seq], dtype=torch.long, device=next(self.parameters()).device)
            logits = self(x)[0, -1] / temperature
            if top_k > 0:
                vals, idx = torch.topk(logits, k=min(top_k, logits.numel()))
                probs = F.softmax(vals, dim=-1)
                pick = idx[torch.multinomial(probs, num_samples=1)].item()
            else:
                probs = F.softmax(logits, dim=-1)
                pick = torch.multinomial(probs, num_samples=1).item()
            seq.append(pick)
            if pick == stoi[SPECIAL_EOS]:
                break
        text = "".join(itos[tok] for tok in seq[1:] if tok not in (stoi[SPECIAL_EOS], stoi[SPECIAL_PAD]))
        return text.strip()


@dataclass
class ArtifactPaths:
    model_pt: Path
    vocab_json: Path
    export_npz: Path


def save_artifacts(model: TinyNameTransformer, vocab: list[str], paths: ArtifactPaths) -> None:
    paths.model_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), paths.model_pt)
    paths.vocab_json.write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    import numpy as np

    np.savez(paths.export_npz, **state)
