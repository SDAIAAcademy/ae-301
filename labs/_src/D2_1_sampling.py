# %% [markdown]
# # Lab 2.1 · Sampling, consistency and invented details
#
# Day 2 · Session 1 (Generative language models) · 25 min. Ghayth will draft replies to customers. A reply that
# changes every time it is generated, or that promises a credit nobody approved, is a problem for Hessa's
# team. You look at one next-token distribution under different temperatures, generate ten replies at three
# settings, and measure two things: how consistent the replies are and how many details they invent.
#
# **Reference state.** `models/ghayth-dec-tiny.pt` (the course decoder) and `data/reply_tasks.jsonl`: ten
# customer messages, each with the facts a reply may use. Nothing from another lab. Replies come from a hosted
# model through `G.chat` (cached in `cache/`); without a working key a scripted stand-in answers, and the
# banner says so. In quick mode (`ANLP_QUICK=1`) the lab uses four tasks, to stay inside a free daily quota.
#
# **Tasks**
# - [ ] 1 · Temperature on one next-token distribution (5')
# - [ ] 2 · Ten replies at three settings (8')
# - [ ] 3 · Consistency across samples (6')
# - [ ] 4 · An invented-detail checker (6')
# - [ ] ✱ Stretch · Fix a seed and see what it does and does not make repeatable

# %%
#@setup
G.banner("Lab 2.1")

# %% [markdown]
# ## 1 · Temperature on one next-token distribution (5')
#
# A decoder ends with one score (a logit) per vocabulary entry. Sampling turns scores into a choice:
# divide by the **temperature**, apply a softmax, optionally keep only the smallest set of tokens whose
# probabilities add up to **top-p** (the nucleus), renormalise, draw.
#
# ✏️ **Exercise** · Write `sampling_distribution(logits, temperature, top_p)`: the probabilities the next token
# is drawn from. Tokens outside the nucleus get probability 0; the rest sum to 1. Keep a token when the
# probability mass before it (in descending order) is below `top_p`, so the top token always stays.

# %%
import json
import re
from itertools import combinations

import numpy as np
import pandas as pd
import torch

import tinytf as T

tok = T.load_tokenizer()
dec = T.load(G.MODELS / "ghayth-dec-tiny.pt")
PREFIX = "الماي"  # "the water" in Gulf dialect
with torch.no_grad():
    LOGITS = dec.lm_logits(dec(T.encode(tok, [PREFIX], cls=False)))[0, -1]


def sampling_distribution(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> torch.Tensor:
    # BEGIN SOLUTION
    probs = (logits / temperature).softmax(-1)
    order = probs.argsort(descending=True)
    before = probs[order].cumsum(0) - probs[order]
    keep = order[before < top_p]
    out = torch.zeros_like(probs)
    out[keep] = probs[keep]
    return out / out.sum()
    # END SOLUTION


# %%
try:
    sampling_distribution(LOGITS)
except NotImplementedError:
    def sampling_distribution(logits, temperature=1.0, top_p=1.0):
        probs = (logits / temperature).softmax(-1)
        order = probs.argsort(descending=True)
        keep = order[(probs[order].cumsum(0) - probs[order]) < top_p]
        out = torch.zeros_like(probs)
        out[keep] = probs[keep]
        return out / out.sum()

rows = []
for t, p in [(0.2, 1.0), (0.7, 1.0), (1.0, 1.0), (1.5, 1.0), (1.0, 0.9), (1.0, 0.5)]:
    d = sampling_distribution(LOGITS, t, p)
    top = d.topk(3)
    rows.append({"temperature": t, "top_p": p, "top token": tok.id_to_token(int(top.indices[0])), "its probability": round(float(top.values[0]), 3),
                 "tokens holding 99%": int((d.sort(descending=True).values.cumsum(0) < 0.99).sum()) + 1, "entropy (bits)": round(float(-(d[d > 0] * d[d > 0].log2()).sum()), 2)})
dist = pd.DataFrame(rows)
print(f"after '{PREFIX}':")
print(dist.to_string(index=False))
assert abs(float(sampling_distribution(LOGITS, 0.7, 0.9).sum()) - 1) < 1e-5
assert dist.iloc[0]["its probability"] > dist.iloc[3]["its probability"], "a low temperature sharpens the distribution"
assert dist.iloc[5]["tokens holding 99%"] < dist.iloc[4]["tokens holding 99%"] < dist.iloc[2]["tokens holding 99%"]

# %% [markdown]
# 🔍 The model is the same in every row. Temperature and top-p change only how its scores are turned into a
# choice. At temperature 0 a provider takes the top token every time (no draw at all).

# %% [markdown]
# ## 2 · Ten replies at three settings (8')
#
# Each task has a customer message and the facts the contact centre knows. The reply may use those facts and
# nothing else.
#
# ✏️ **Exercise** · Write `draft(task, temperature, top_p, sample)`: one call to `G.chat` with `SYSTEM` and the
# user message below, `max_tokens=120`, and `sample=sample` (it keeps repeated samples apart in the cache).
# Return the reply text.

# %%
TASKS = G.jsonl("reply_tasks.jsonl")[: G.pick(10, 4)]
SETTINGS = [("focused", 0.0, 1.0), ("balanced", 0.7, 0.9), ("loose", 1.3, 1.0)]
N_SAMPLES = 2
SYSTEM = ("You write short replies for the contact centre of Sawaqi Water Company. Reply in the customer's language, "
          "in at most three sentences. Use only the facts given. Do not promise anything that is not in the facts.")


def user_message(task: dict) -> str:
    return f"Customer message: {task['message']}\nFACTS: {json.dumps(task['facts'], ensure_ascii=False)}"


def draft(task: dict, temperature: float, top_p: float, sample: int) -> str:
    # BEGIN SOLUTION
    r = G.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(task)}],
               temperature=temperature, top_p=top_p, max_tokens=120, sample=sample)
    return r["text"].strip()
    # END SOLUTION


# %%
def all_replies():
    return [{"task": t["id"], "setting": name, "sample": s, "reply": draft(t, temp, p, s)}
            for t in TASKS for name, temp, p in SETTINGS for s in range(N_SAMPLES)]


replies = pd.DataFrame(G.attempt(all_replies, "d21_replies"))
replies = replies[replies["task"].isin([t["id"] for t in TASKS])]  # the reference result covers all ten tasks
print(len(replies), "replies |", "stand-in (scripted, not a model)" if G.standin() else G.CHAT_MODEL)
for _, r in replies[replies["task"] == replies["task"].iloc[0]].iterrows():
    print(f"  {r['setting']:9s}#{r['sample']}  {r['reply']}")
assert len(replies) == len(TASKS) * len(SETTINGS) * N_SAMPLES and replies["reply"].str.len().min() > 0

# %% [markdown]
# ## 3 · Consistency across samples (6')
#
# Two samples of the same task at the same setting should say the same thing. Two simple measures: **exact
# agreement** (the share of pairs with identical text) and **word overlap** (the mean Jaccard similarity of
# the pairs' word sets: shared words divided by all words).
#
# ✏️ **Exercise** · Write `consistency(texts)` for the samples of one task at one setting: a dictionary with
# `exact` and `overlap`, each the mean over all pairs.

# %%
def consistency(texts: list[str]) -> dict:
    # BEGIN SOLUTION
    exact, overlap = [], []
    for a, b in combinations(texts, 2):
        wa, wb = set(a.lower().split()), set(b.lower().split())
        exact.append(float(a.strip() == b.strip()))
        overlap.append(len(wa & wb) / max(1, len(wa | wb)))
    return {"exact": float(np.mean(exact)), "overlap": float(np.mean(overlap))}
    # END SOLUTION


# %%
try:
    assert consistency(["a b c", "a b c"]) == {"exact": 1.0, "overlap": 1.0}
    assert consistency(["a b c d", "a b x y"])["overlap"] == 2 / 6
except NotImplementedError:
    def consistency(texts):
        pairs = list(combinations(texts, 2))
        sets = [(set(a.lower().split()), set(b.lower().split())) for a, b in pairs]
        return {"exact": float(np.mean([a.strip() == b.strip() for a, b in pairs])),
                "overlap": float(np.mean([len(x & y) / max(1, len(x | y)) for x, y in sets]))}

cons = replies.groupby(["setting", "task"], sort=False)["reply"].apply(lambda s: pd.Series(consistency(list(s)))).unstack()
cons = cons.groupby("setting", sort=False).mean().round(3)
print(cons.to_string())

# %% [markdown]
# ## 4 · An invented-detail checker (6')
#
# A detail is **invented** when the reply states it and neither the customer's message nor the facts hold it.
# Checking meaning needs a person or a second model; checking **numbers and identifiers** needs a regular
# expression, and those are the details that cost money: an amount, a number of hours, a ticket, a rule.
#
# ✏️ **Exercise** · Write `invented(reply, task)`: the list of numbers (runs of digits) and identifiers
# (`SW-` or `R-` followed by digits) in the reply that appear in neither the message nor the facts.

# %%
def invented(reply: str, task: dict) -> list[str]:
    # BEGIN SOLUTION
    allowed = task["message"] + " " + json.dumps(task["facts"], ensure_ascii=False)
    found = re.findall(r"(?:SW|R)-\d+|\d+(?:\.\d+)?", reply)
    return [d for d in found if d not in allowed]
    # END SOLUTION


# %%
by_id = {t["id"]: t for t in TASKS}
try:
    assert invented("Ticket SW-48211, a crew within 4 hours.", by_id["r01"]) == []
    assert invented("Ticket SW-48211. You get 50 SAR under R-11.", by_id["r01"]) == ["50", "R-11"]
except NotImplementedError:
    def invented(reply, task):
        allowed = task["message"] + " " + json.dumps(task["facts"], ensure_ascii=False)
        return [d for d in re.findall(r"(?:SW|R)-\d+|\d+(?:\.\d+)?", reply) if d not in allowed]

replies["invented"] = [invented(r, by_id[t]) for r, t in zip(replies["reply"], replies["task"])]
summary = cons.join(replies.groupby("setting", sort=False)["invented"].apply(lambda s: int(sum(bool(x) for x in s))).rename("replies_with_invented_detail"))
summary["replies"] = replies.groupby("setting", sort=False).size()
print(summary.to_string())
print()
print(replies[replies["invented"].astype(bool)][["task", "setting", "invented", "reply"]].head(6).to_string(index=False))
out = G.lab_dir("D2_1")
summary.to_csv(out / "sampling_summary.csv")
G.keep("d21_summary", summary.reset_index().to_dict(orient="records"))

# %% [markdown]
# 🔍 Read one invented detail in full. Would a customer act on it? The checker sees numbers and identifiers
# only: an invented name or an invented promise in words passes it. What would you add for those?

# %% [markdown]
# ## ✱ Stretch · Fix a seed and see what it does and does not make repeatable
#
# `G.chat(..., seed_=7)` passes a seed to the provider. Generate the first task four times at the loose
# setting with the same seed and with `ANLP_CACHE=0`. Are the replies identical? Read your provider's page on
# the seed parameter before you rely on it: many say "best effort".

# %%
# BEGIN SOLUTION
seeded = [G.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(TASKS[0])}],
                 temperature=1.3, max_tokens=120, sample=100 + i, seed_=7)["text"] for i in range(4)]
print(consistency(seeded))
# END SOLUTION

# %% [markdown]
# ## Checkpoint
#
# 1. Temperature 0 made the replies repeat. Did it make them right?
# 2. Why keep `top_p` below 1 when the temperature is already moderate?
# 3. The routing label of the same message changes from run to run. What do you change first?
#
# <details><summary>Answers</summary>
#
# 1. No. It makes the model say its most likely reply every time, right or wrong. A repeated invented detail
#    is still invented.
# 2. It removes the long tail: thousands of unlikely tokens that together still hold some probability, and
#    that produce the odd word when one of them is drawn.
# 3. The sampling settings: temperature 0 for a decision. Then constrain the output to the eight labels.
#
# </details>
#
# **Carry forward.** `artifacts/D2_1/sampling_summary.csv`. Lab 2.2 uses temperature 0 for extraction.
