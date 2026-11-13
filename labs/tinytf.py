"""The course's own tiny transformer: an encoder and a decoder small enough for a laptop processor.

Both have the same parts as a large model (token and position embeddings, multi-head attention
with query, key, value and output projections, a feed-forward block, layer norm, residuals), at
a size that trains in a minute. The encoder lets every token see every other token; the decoder
masks the future. `labs/_tools/pretrain_tiny.py` trained both on the training split of the Sawaqi
messages and the rule book; the weights are in `labs/models/`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS = Path(__file__).resolve().parent / "models"
SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]


@dataclass
class Config:
    vocab: int = 2000
    d: int = 128
    heads: int = 4
    layers: int = 2
    ff: int = 256
    max_len: int = 64
    causal: bool = False
    dropout: float = 0.1


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.h, self.dk, self.causal = cfg.heads, cfg.d // cfg.heads, cfg.causal
        self.q, self.k, self.v, self.o = (nn.Linear(cfg.d, cfg.d) for _ in range(4))
        self.drop = nn.Dropout(cfg.dropout)
        self.last = None  # attention weights of the last call: (batch, heads, query, key)

    def forward(self, x, pad_mask):
        b, t, _ = x.shape
        split = lambda y: y.view(b, t, self.h, self.dk).transpose(1, 2)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)
        mask = pad_mask[:, None, None, :]  # True where a key is a real token
        if self.causal:
            mask = mask & torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
        w = scores.softmax(-1)
        self.last = w.detach()
        y = (self.drop(w) @ v).transpose(1, 2).reshape(b, t, -1)
        return self.o(y)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d), nn.LayerNorm(cfg.d)
        self.attn = Attention(cfg)
        self.ff = nn.Sequential(nn.Linear(cfg.d, cfg.ff), nn.GELU(), nn.Linear(cfg.ff, cfg.d))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, pad_mask):
        x = x + self.drop(self.attn(self.ln1(x), pad_mask))
        return x + self.drop(self.ff(self.ln2(x)))


class TinyTransformer(nn.Module):
    """Returns hidden states (batch, tokens, d); `lm_logits` maps them to the vocabulary."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d, padding_idx=0)
        self.pos = nn.Embedding(cfg.max_len, cfg.d)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.ln = nn.LayerNorm(cfg.d)

    def forward(self, ids, pad_mask=None):
        if pad_mask is None:
            pad_mask = ids != 0
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.tok(ids) + self.pos(pos)[None]
        for blk in self.blocks:
            x = blk(x, pad_mask)
        return self.ln(x)

    def lm_logits(self, hidden):
        return hidden @ self.tok.weight.T  # tied with the input embeddings

    def attentions(self):
        """One (batch, heads, query, key) tensor per layer, from the last forward pass."""
        return [blk.attn.last for blk in self.blocks]


class Classifier(nn.Module):
    """An encoder with a classification head on the mean of its token states."""

    def __init__(self, encoder: TinyTransformer, n_labels: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(encoder.cfg.d, n_labels))

    def forward(self, ids):
        mask = ids != 0
        h = self.encoder(ids, mask)
        pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.head(pooled)


def load_tokenizer(path: Path | None = None):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path or MODELS / "ghayth-bpe.json"))


def encode(tokenizer, texts, max_len: int = 64, cls: bool = True) -> torch.Tensor:
    """Token ids, padded with 0 and cut at max_len; [CLS] first when cls is True."""
    cls_id = tokenizer.token_to_id("[CLS]")
    rows = []
    for t in texts:
        ids = tokenizer.encode(t).ids
        ids = ([cls_id] + ids) if cls else ids
        rows.append(ids[:max_len])
    width = max(1, max(len(r) for r in rows))
    return torch.tensor([r + [0] * (width - len(r)) for r in rows], dtype=torch.long)


def mask_tokens(ids: torch.Tensor, mask_id: int, p: float = 0.15) -> torch.Tensor:
    """A copy of the batch with a share p of its real tokens replaced by [MASK]: cheap augmentation when labels are few."""
    out = ids.clone()
    out[(torch.rand(ids.shape) < p) & (ids != 0)] = mask_id
    return out


def save(model: nn.Module, path: Path, **extra) -> None:
    cfg = model.cfg if hasattr(model, "cfg") else model.encoder.cfg
    torch.save({"config": asdict(cfg), "state": model.state_dict(), **extra}, path)


def load(path: Path) -> TinyTransformer:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    model = TinyTransformer(Config(**blob["config"]))
    model.load_state_dict(blob["state"])
    return model.eval()


def load_classifier(path: Path) -> tuple[Classifier, list[str]]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    labels = blob["labels"].split(",")
    model = Classifier(TinyTransformer(Config(**blob["config"])), len(labels))
    model.load_state_dict(blob["state"])
    return model.eval(), labels


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


# ---------- reference versions of Lab 3.3's exercises (the lab's safety nets use them) ----------

class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank update: y = W x + (alpha / r) B A x."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base, self.scale = base, alpha / r
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


def add_lora(model: Classifier, r: int = 8, alpha: int = 16) -> Classifier:
    """Freezes the encoder and wraps the query and value projections of every attention block."""
    for p in model.encoder.parameters():
        p.requires_grad = False
    for blk in model.encoder.blocks:
        blk.attn.q, blk.attn.v = LoRALinear(blk.attn.q, r, alpha), LoRALinear(blk.attn.v, r, alpha)
    return model


def quantise_int8(w: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Symmetric 8-bit quantisation of one tensor: integers in [-127, 127] and one scale."""
    scale = float(w.abs().max()) / 127 or 1.0
    return (w / scale).round().clamp(-127, 127).to(torch.int8), scale
