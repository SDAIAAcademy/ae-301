# %% [markdown]
# # Lab 3.3 · A low-rank adapter in a memory budget
#
# Day 3 · Session 3 (Low-rank and quantised adapters) · 25 min. Full fine-tuning changes every weight, and
# training needs room for every weight three more times: its gradient and two optimiser states. A low-rank
# adapter freezes the model and trains two small matrices beside a few of its layers; the frozen weights can
# then be held in fewer bits. You write the adapter layer, count what trains, do the memory arithmetic, and
# compare your adapter run with a full fine-tuning run that the course supplies.
#
# **Reference state.** `models/ghayth-enc-tiny.pt` (the pretrained course encoder), the split in
# `data/split.csv`, and the supplied full fine-tuning run: its log and test scores in
# `data/reference/full_ft_log.json`, its weights in `models/ghayth-enc-routing-full.pt`. Nothing from another
# lab. Everything runs on a laptop processor in about a minute; the same code runs on a notebook GPU.
#
# **Tasks**
# - [ ] 1 · A low-rank adapter layer by hand (7')
# - [ ] 2 · Wrap the attention and count parameters (4')
# - [ ] 3 · A memory budget, with 8-bit frozen weights (6')
# - [ ] 4 · Train and compare with full fine-tuning (8')
# - [ ] ✱ Stretch · Sweep the rank and plot macro F1 against trainable parameters

# %%
#@setup
G.banner("Lab 3.3")

# %%
import json
import time
from datetime import datetime, timezone

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import tinytf as T

G.seed()
tok = T.load_tokenizer()
LABELS = G.DEPARTMENTS
train, val, test = G.messages("train"), G.messages("val"), G.messages("test")


def tensors(df):
    return T.encode(tok, list(df["text"])), torch.tensor([LABELS.index(d) for d in df["department"]])


(xtr, ytr), (xva, yva), (xte, yte) = tensors(train), tensors(val), tensors(test)
FULL = G.reference("full_ft_log")
print("supplied full fine-tuning run:", FULL["trainable_params"], "trainable parameters, test macro F1", FULL["test"][0]["macro_f1"])

# %% [markdown]
# ## 1 · A low-rank adapter layer by hand (7')
#
# A linear layer computes `W x + b`. Fine-tuning would change `W` by some `ΔW` of the same size. The low-rank
# idea: write `ΔW = B A`, with `A` of shape (r, inputs) and `B` of shape (outputs, r), and `r` small. Train
# only `A` and `B`. Start `B` at zero, so the adapted layer begins exactly as the frozen one. Scale the
# update by `alpha / r`.
#
# ✏️ **Exercise** · Complete `LoRALinear`: freeze the base layer's parameters; create `A` (normal, standard
# deviation 0.01) and `B` (zeros) as `nn.Parameter`; in `forward`, return the base output plus the scaled
# low-rank update.

# %%
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base, self.scale = base, alpha / r
        # BEGIN SOLUTION
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        # END SOLUTION

    def forward(self, x):
        # BEGIN SOLUTION
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale
        # END SOLUTION


# %%
try:
    LoRALinear(nn.Linear(4, 4))(torch.zeros(1, 4))
except NotImplementedError:
    LoRALinear = T.LoRALinear  # the reference version, so the cells below run

base = nn.Linear(128, 128)
layer = LoRALinear(base, r=8, alpha=16)
x = torch.randn(3, 128)
assert torch.allclose(layer(x), base(x)), "with B at zero the adapted layer must equal the frozen one"
assert [n for n, p in layer.named_parameters() if p.requires_grad] == ["A", "B"], "only A and B train"
layer.B.data.normal_()
assert not torch.allclose(layer(x), base(x)) and layer.A.shape == (8, 128) and layer.B.shape == (128, 8)
print("✅ LoRALinear: starts equal to the frozen layer; trains", layer.A.numel() + layer.B.numel(), "numbers beside", base.weight.numel())

# %% [markdown]
# ## 2 · Wrap the attention and count parameters (4')
#
# The usual place for adapters is the **query** and **value** projections of every attention block. The
# classification head is new, so it trains in full.
#
# ✏️ **Exercise** · Write `add_lora(model, r, alpha)`: freeze every parameter of `model.encoder`, then replace
# `blk.attn.q` and `blk.attn.v` of every block by a `LoRALinear` around the old layer. Return the model.

# %%
def add_lora(model: T.Classifier, r: int = 8, alpha: int = 16) -> T.Classifier:
    # BEGIN SOLUTION
    for p in model.encoder.parameters():
        p.requires_grad = False
    for blk in model.encoder.blocks:
        blk.attn.q, blk.attn.v = LoRALinear(blk.attn.q, r, alpha), LoRALinear(blk.attn.v, r, alpha)
    return model
    # END SOLUTION


# %%
def fresh(r: int = 8) -> T.Classifier:
    G.seed()
    model = T.Classifier(T.load(G.MODELS / "ghayth-enc-tiny.pt"), len(LABELS))
    try:
        return add_lora(model, r, 2 * r)
    except NotImplementedError:
        return T.add_lora(model, r, 2 * r)


model = fresh(8)
total, trainable = T.count_params(model), T.count_params(model, trainable_only=True)
print(f"total {total:,} | trainable {trainable:,} ({trainable / total:.1%}) | full fine-tuning trains {FULL['trainable_params']:,}")
assert trainable == 4 * (8 * 128 * 2) + (128 * len(LABELS) + len(LABELS)), "four adapted layers and the head"

# %% [markdown]
# ## 3 · A memory budget, with 8-bit frozen weights (6')
#
# What training keeps in memory for the weights (activations come on top, and depend on the batch):
#
# - a **trainable** weight: 4 bytes for the value, 4 for its gradient, 8 for the two states of Adam: 16 bytes;
# - a **frozen** weight: its value only: 4 bytes in 32-bit floats, 1 byte in 8-bit integers, half a byte in 4 bits.
#
# ✏️ **Exercise** · Write `training_megabytes(total, trainable, frozen_bits)`: megabytes (10^6 bytes) for a
# model with `total` parameters of which `trainable` train, with the frozen ones held in `frozen_bits` bits.

# %%
def training_megabytes(total: int, trainable: int, frozen_bits: int = 32) -> float:
    # BEGIN SOLUTION
    return ((total - trainable) * frozen_bits / 8 + trainable * 16) / 1e6
    # END SOLUTION


# %%
try:
    training_megabytes(10, 10)
except NotImplementedError:
    def training_megabytes(total, trainable, frozen_bits=32):
        return ((total - trainable) * frozen_bits / 8 + trainable * 16) / 1e6

assert training_megabytes(1_000_000, 1_000_000) == 16.0 and training_megabytes(1_000_000, 0, 8) == 1.0
BUDGET_MB = 3.0  # the budget of this lab, for the course encoder
rows = []
for name, n_total, n_train, bits in [("full fine-tuning", FULL["total_params"], FULL["trainable_params"], 32),
                                     ("adapter, 32-bit frozen weights", total, trainable, 32),
                                     ("adapter, 8-bit frozen weights", total, trainable, 8)]:
    mb = training_megabytes(n_total, n_train, bits)
    rows.append({"run": name, "trainable": n_train, "megabytes": round(mb, 2), "inside the 3 MB budget": mb <= BUDGET_MB})
budget = pd.DataFrame(rows)
print(budget.to_string(index=False))
print("\nThe same arithmetic at other sizes (gigabytes; adapters at 0.5% of the weights):")
for name, n in [("an encoder of 110 million parameters", 110e6), ("a decoder of 7 billion parameters", 7e9)]:
    print(f"  {name}: full {training_megabytes(n, n) / 1e3:.1f} | adapter, 32-bit {training_megabytes(n, 0.005 * n) / 1e3:.1f}"
          f" | adapter, 4-bit {training_megabytes(n, 0.005 * n, 4) / 1e3:.1f}")
assert list(budget["inside the 3 MB budget"]) == [False, True, True]

# %% [markdown]
# 🔍 Check the memory of the GPU your notebook service gives you today, and read the last line again: which
# of the three ways of training a 7-billion-parameter model fits? Quantising is not free: the next cell
# rounds the frozen weights of the course encoder to 8 bits and shows how far they move.

# %%
def quantise_frozen(model: T.Classifier) -> float:
    """Rounds every frozen weight matrix to 8-bit integers and back; returns the mean relative error."""
    errors = []
    for p in model.encoder.parameters():
        if not p.requires_grad and p.dim() == 2:
            q, scale = T.quantise_int8(p.data)
            back = q.float() * scale
            errors.append(float((back - p.data).abs().mean() / p.data.abs().mean()))
            p.data.copy_(back)
    return sum(errors) / len(errors)


err = quantise_frozen(model)
print(f"frozen weights rounded to 8 bits: mean relative error {err:.2%}")
assert 0 < err < 0.1

# %% [markdown]
# ## 4 · Train and compare with full fine-tuning (8')
#
# ✏️ **Exercise** · Write `train_adapter(model, epochs, lr)`. Give AdamW only the parameters that require a
# gradient. Each epoch: shuffle, batches of 32, mask 15% of the tokens of each batch with
# `T.mask_tokens(batch, MASK_ID)` as the full run did, cross-entropy, step; then compute the validation macro F1
# with `macro_f1(model, xva, yva)` and append a log entry with `epoch`, `train_loss`, `val_macro_f1` and a
# UTC `timestamp`. Return the log.

# %%
def predict(model, x) -> list[str]:
    model.eval()
    with torch.no_grad():
        return [LABELS[i] for i in model(x).argmax(-1).tolist()]


def macro_f1(model, x, y) -> float:
    return float(G.by_variety([LABELS[i] for i in y.tolist()], predict(model, x), [None] * len(y)).iloc[0]["macro_f1"])


MASK_ID = tok.token_to_id("[MASK]")


def train_adapter(model, epochs: int, lr: float = 5e-3) -> list[dict]:
    # BEGIN SOLUTION
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    log = []
    for ep in range(epochs):
        model.train()
        perm, total_loss = torch.randperm(len(xtr)), 0.0
        for s in range(0, len(xtr), 32):
            idx = perm[s:s + 32]
            loss = F.cross_entropy(model(T.mask_tokens(xtr[idx], MASK_ID)), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        log.append({"epoch": ep + 1, "train_loss": round(total_loss / len(xtr), 4), "val_macro_f1": macro_f1(model, xva, yva),
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return log
    # END SOLUTION


# %%
EPOCHS = G.pick(40, 4)
t0 = time.perf_counter()
try:
    log = train_adapter(model, EPOCHS)
    took = time.perf_counter() - t0
    table = G.by_variety(test["department"], predict(model, xte), test["variety"])
    run = {"method": "low-rank adapter (r=8) on 8-bit frozen weights", "trainable_params": trainable, "total_params": total,
           "epochs": EPOCHS, "log": log, "test": table.to_dict(orient="records"), "seconds": round(took, 1)}
    G.keep("lora_log", run)
except NotImplementedError:
    run = G.reference("lora_log")
    print("↩️  train_adapter not written yet: using the reference adapter run")
print(pd.DataFrame(run["log"])[["epoch", "train_loss", "val_macro_f1"]].iloc[4::5].to_string(index=False))

compare = pd.DataFrame([{"run": r["method"], "trainable": r["trainable_params"],
                         **{row["variety"]: row["macro_f1"] for row in r["test"]}} for r in (FULL, run)]).rename(columns={"all": "macro_f1"})
print()
print(compare.to_string(index=False))
out = G.lab_dir("D3_3")
(out / "adapter_run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
compare.to_csv(out / "adapter_vs_full.csv", index=False)
assert len(run["log"]) >= 2 and run["log"][-1]["train_loss"] < run["log"][0]["train_loss"]

# %% [markdown]
# 🔍 This table, with both logs, is the evidence of this session: trainable parameters and test scores for
# both runs. What did the adapter give up, in which variety, for what saving? At this size the saving is
# megabytes; do the same division for the 7-billion-parameter line of step 3.

# %% [markdown]
# ## ✱ Stretch · Sweep the rank and plot macro F1 against trainable parameters
#
# Train adapters of rank 2, 4, 8 and 16 for 15 epochs each and plot test macro F1 against trainable
# parameters, with the full fine-tuning run as a point. Where does the curve flatten?

# %%
# BEGIN SOLUTION
points = []
for r in (2, 16) if G.QUICK else (2, 4, 8, 16):
    m = fresh(r)
    train_adapter(m, G.pick(15, 2))
    points.append({"rank": r, "trainable": T.count_params(m, trainable_only=True), "test_macro_f1": macro_f1(m, xte, yte)})
print(pd.DataFrame(points).to_string(index=False))
# END SOLUTION

# %% [markdown]
# ## Checkpoint
#
# 1. Why does `B` start at zero and `A` not?
# 2. The tariff changes every quarter. Should Ghayth learn it through an adapter?
# 3. After training, how do you serve an adapter without extra latency?
#
# <details><summary>Answers</summary>
#
# 1. With `B` at zero the update `B A` is zero, so training starts from the pretrained model's behaviour. If
#    both were zero, neither would receive a gradient; `A` is random so that `B` can start to learn.
# 2. No. Facts that change belong in the rule book that Lab 2.3 retrieves from: updating a file takes
#    minutes and is auditable. Adapters are for a skill (routing in Sawaqi's four varieties), not for prices.
# 3. Merge it: add `(alpha / r) B A` into the frozen weight once. The layer is then an ordinary linear layer.
#
# </details>
#
# **Carry forward.** `artifacts/D3_3/adapter_vs_full.csv` and `adapter_run.json` (with timestamps): the
# evidence for the adaptation criterion of the capstone.
