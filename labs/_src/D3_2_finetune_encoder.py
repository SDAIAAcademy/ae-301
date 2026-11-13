# %% [markdown]
# # Lab 3.2 · Fine-tune the course encoder for routing
#
# Day 3 · Session 2 (Fine-tuning an encoder) · 25 min. The third way to route: take an encoder that has
# already read Sawaqi's archive without labels, put a small classification head on it, and train both on the
# labelled messages. You write the head and the training loop, pick the epoch on the validation split, score
# the test split by variety and read the confusion matrix.
#
# **Reference state.** `models/ghayth-enc-tiny.pt` (the course encoder: small, Arabic-capable because it was
# pretrained on these messages in four varieties, without labels), `models/ghayth-bpe.json` and the split in
# `data/split.csv`. The baseline's predictions come from your Lab 1.4 or from `data/reference/`. Nothing is
# downloaded; training takes about two minutes on a laptop processor (seconds in quick mode).
#
# **Tasks**
# - [ ] 1 · Load the encoder and encode a batch (4')
# - [ ] 2 · A classification head and a training loop (9')
# - [ ] 3 · Macro F1 overall and by variety (5')
# - [ ] 4 · Read the confusion matrix (7')
# - [ ] ✱ Stretch · Run the same notebook on a published Arabic encoder in Colab with a GPU

# %%
#@setup
G.banner("Lab 3.2")

# %% [markdown]
# ## 1 · Load the encoder and encode a batch (4')

# %%
import copy
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
encoder = T.load(G.MODELS / "ghayth-enc-tiny.pt")
with torch.no_grad():
    hidden = encoder(xtr[:4])
print("token ids:", tuple(xtr[:4].shape), "→ hidden states:", tuple(hidden.shape), "| 0 is padding")
print("train", len(xtr), "| val", len(xva), "| test", len(xte), "| encoder parameters:", f"{T.count_params(encoder):,}")

# %% [markdown]
# ## 2 · A classification head and a training loop (9')
#
# The encoder returns one vector per token. A message needs one vector: the **mean of the real tokens**
# (padding left out). A linear layer turns that vector into eight scores.
#
# ✏️ **Exercise** · Complete `Router.forward`: run the encoder, average the hidden states over the positions
# where `ids != 0`, apply dropout and the head. Then write `train_epoch(model, opt, x, y)`: one pass over the
# shuffled data in batches of 32 with cross-entropy; return the mean loss.

# %%
class Router(nn.Module):
    def __init__(self, encoder: T.TinyTransformer, n_labels: int):
        super().__init__()
        self.encoder = encoder
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(encoder.cfg.d, n_labels)

    def forward(self, ids):
        # BEGIN SOLUTION
        mask = ids != 0
        h = self.encoder(ids, mask)
        pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.head(self.drop(pooled))
        # END SOLUTION


def train_epoch(model, opt, x, y) -> float:
    # BEGIN SOLUTION
    model.train()
    perm, total = torch.randperm(len(x)), 0.0
    for s in range(0, len(x), 32):
        idx = perm[s:s + 32]
        loss = F.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    return total / len(x)
    # END SOLUTION


# %%
def predict(model, x) -> list[str]:
    model.eval()
    with torch.no_grad():
        return [LABELS[i] for i in model(x).argmax(-1).tolist()]


def macro_f1(model, x, y) -> float:
    return float(G.by_variety([LABELS[i] for i in y.tolist()], predict(model, x), [None] * len(y)).iloc[0]["macro_f1"])


MASK_ID = tok.token_to_id("[MASK]")


def fine_tune(epochs: int, lr: float = 1e-3):
    """Trains every weight; keeps the weights of the epoch with the best validation macro F1.

    In every epoch 15% of the input tokens are replaced by [MASK], as in the encoder's pretraining:
    with 400 labelled messages the model would otherwise learn single words by heart.
    """
    G.seed()
    model = Router(T.load(G.MODELS / "ghayth-enc-tiny.pt"), len(LABELS))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    log, best, best_state = [], -1.0, None
    for ep in range(epochs):
        loss = train_epoch(model, opt, T.mask_tokens(xtr, MASK_ID), ytr)
        f1 = macro_f1(model, xva, yva)
        log.append({"epoch": ep + 1, "train_loss": round(loss, 4), "val_macro_f1": f1,
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        if f1 > best:
            best, best_state = f1, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, log


EPOCHS = G.pick(40, 4)
t0 = time.perf_counter()
try:
    model, log = fine_tune(EPOCHS)
    y_pred = predict(model, xte)
    G.keep("d32_log", log)
except NotImplementedError:
    print("↩️  Exercise not done yet: using the supplied fine-tuned encoder and its log")
    model, _ = T.load_classifier(G.MODELS / "ghayth-enc-routing-full.pt")
    log = G.reference("d32_log")
    y_pred = predict(model, xte)
print(pd.DataFrame(log)[["epoch", "train_loss", "val_macro_f1"]].iloc[4::5].to_string(index=False))
best_epoch = max(log, key=lambda e: e["val_macro_f1"])["epoch"]
print(f"best validation macro F1 at epoch {best_epoch} of {len(log)} | {time.perf_counter() - t0:.0f} s on this processor")
assert log[-1]["train_loss"] < log[0]["train_loss"]

# %% [markdown]
# 🔍 The training loss keeps falling after the validation score has stopped rising. The loop kept the weights
# of the best validation epoch. Why is it the validation split that chooses, and not the test split?

# %% [markdown]
# ## 3 · Macro F1 overall and by variety (5')
#
# ✏️ **Exercise** · Build `scores`: a dictionary from `"baseline"` and `"fine-tuned encoder"` to the macro F1
# overall and by variety on the test split, as `G.by_variety(...)` returns it, turned into a dictionary from
# variety to macro F1. The baseline's predictions are in `baseline`.

# %%
baseline = pd.read_csv(G.supplied("baseline_predictions.csv", "D1_4")).set_index("msg_id").loc[test["msg_id"]]


def variety_scores(y_pred) -> dict:
    return G.by_variety(test["department"], y_pred, test["variety"]).set_index("variety")["macro_f1"].to_dict()


# BEGIN SOLUTION
scores = {"baseline": variety_scores(baseline["y_pred"]), "fine-tuned encoder": variety_scores(y_pred)}
# END SOLUTION

# %%
try:
    scores
except NameError:
    scores = {"baseline": variety_scores(baseline["y_pred"]), "fine-tuned encoder": variety_scores(y_pred)}
table = pd.DataFrame(scores).T.rename(columns={"all": "macro_f1"})
print(table.to_string())
out = G.lab_dir("D3_2")
table.to_csv(out / "encoder_vs_baseline.csv")
predictions = pd.DataFrame({"msg_id": test["msg_id"], "variety": test["variety"], "y_true": test["department"], "y_pred": y_pred})
predictions.to_csv(out / "encoder_predictions.csv", index=False)
(out / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
G.keep_file(out / "encoder_predictions.csv")
G.keep("d32_scores", scores)
assert table.loc["fine-tuned encoder", "macro_f1"] > (0.1 if G.QUICK else 0.5)

# %% [markdown]
# 🔍 Read the table by column, not only the first one. Where did 400 labels and a pretrained encoder beat
# word counts, and where not? This is the third row of the comparison the course asks you to defend.

# %% [markdown]
# ## 4 · Read the confusion matrix (7')
#
# A confusion matrix counts, for every true department, where the messages went. The diagonal is right;
# every other cell is a kind of error with a cause you can look for.
#
# ✏️ **Exercise** · Write `top_confusions(y_true, y_pred, k)`: the `k` largest off-diagonal cells as a list of
# `(true, predicted, count)`, largest first.

# %%
def top_confusions(y_true, y_pred, k: int = 5) -> list:
    # BEGIN SOLUTION
    pairs = pd.Series(list(zip(y_true, y_pred)))
    counts = pairs[[t != p for t, p in pairs]].value_counts()
    return [(t, p, int(n)) for (t, p), n in counts.head(k).items()]
    # END SOLUTION


# %%
print(pd.crosstab(test["department"], pd.Series(y_pred, name="predicted", index=test.index)).reindex(index=LABELS, columns=LABELS, fill_value=0).to_string())
confusions = [tuple(c) for c in G.attempt(lambda: top_confusions(list(test["department"]), y_pred), "d32_confusions")]
print("\nlargest confusions (true → predicted):")
for t, p, n in confusions:
    print(f"  {t} → {p}: {n}")
t, p, _ = confusions[0]
wrong = test[(test["department"] == t) & (pd.Series(y_pred, index=test.index) == p)]
print(f"\nthree '{t}' messages sent to '{p}':")
for text in wrong["text"].head(3):
    print("  ", text[:150])
assert len(confusions) >= 3 and all(a != b for a, b, _ in confusions)

# %% [markdown]
# 🔍 Read the three messages. Do they carry a second clause with another department's words ("and the meter
# was replaced last month")? Is "my bill is high and I think the meter is broken" a billing message or a
# meters message? Some confusions are the model's; some are the label's. Say which, before you ask for more
# training.

# %% [markdown]
# ## ✱ Stretch · Run the same notebook on a published Arabic encoder in Colab with a GPU
#
# The course encoder knows Sawaqi's messages and nothing else. A published Arabic encoder has read far more
# text. In Colab, with a GPU runtime, install `transformers`, load an Arabic-capable encoder of your choice
# with `AutoModel.from_pretrained` and its tokeniser, replace `Router.encoder` by it (its hidden states are
# `model(**batch).last_hidden_state`; the mask is `batch["attention_mask"]`), lower the learning rate to
# about 2e-5 and train for three epochs. Compare the same table. This step downloads model weights: do it in
# Colab, not on the classroom laptops.

# %% [markdown]
# ## Checkpoint
#
# 1. Validation macro F1 peaked at an early epoch and then moved sideways. What do you ship?
# 2. Why average only over real tokens, and not over the padded positions too?
# 3. The largest confusion is between two departments whose messages share words. Name two fixes that are
#    not "train longer".
#
# <details><summary>Answers</summary>
#
# 1. The weights of the best validation epoch. Later epochs fit the training messages more closely without
#    doing better on unseen ones.
# 2. Padding carries no meaning; averaging it in would make a short message's vector depend on how long the
#    longest message of its batch happened to be.
# 3. Fix the labels (decide where a "high bill, maybe the meter" message belongs and relabel consistently),
#    add labelled examples of exactly that kind, or send that pair to a person when the two scores are close.
#
# </details>
#
# **Carry forward.** `artifacts/D3_2/encoder_predictions.csv`, `encoder_vs_baseline.csv` and `train_log.json`
# (with timestamps). Lab 3.4 serves a routing model and writes its card from these numbers.
