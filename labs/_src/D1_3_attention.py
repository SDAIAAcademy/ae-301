# %% [markdown]
# # Lab 1.3 · Attention in an encoder and a decoder
#
# Day 1 · Session 3 (Inside the transformer) · 25 min. You open two small transformers that were trained on
# Sawaqi's messages: an encoder (it fills gaps) and a decoder (it predicts the next token). Same parts, one
# difference: the decoder may not look ahead. You compute attention by hand, read the maps of one Arabic
# sentence in both, and compare what each can predict.
#
# **Reference state.** `models/ghayth-enc-tiny.pt` and `models/ghayth-dec-tiny.pt` (about 0.7 million
# parameters each, two layers of four heads, trained by the course team on a laptop processor; the script is
# `_tools/pretrain_tiny.py`) and their tokeniser `models/ghayth-bpe.json`. They are teaching models: they
# know Sawaqi's messages and nothing else. Nothing from another lab; nothing is downloaded.
#
# **Tasks**
# - [ ] 1 · Load the course encoder and decoder (4')
# - [ ] 2 · Attention by hand for one head (7')
# - [ ] 3 · Read the maps: who attends to whom (7')
# - [ ] 4 · Fill the gap, or predict the next token (7')
# - [ ] ✱ Stretch · Change one letter of the sentence and compare the maps

# %%
#@setup
G.banner("Lab 1.3")

# %% [markdown]
# ## 1 · Load the course encoder and decoder (4')

# %%
import math

import numpy as np
import pandas as pd
import torch

import tinytf as T

tok = T.load_tokenizer()
enc = T.load(G.MODELS / "ghayth-enc-tiny.pt")
dec = T.load(G.MODELS / "ghayth-dec-tiny.pt")
print(f"encoder: {T.count_params(enc):,} parameters, causal={enc.cfg.causal} | decoder: {T.count_params(dec):,}, causal={dec.cfg.causal}")
print(f"layers {enc.cfg.layers}, heads {enc.cfg.heads}, width {enc.cfg.d}, vocabulary {enc.cfg.vocab}")

SENT = "العداد واقف مع اننا نستخدم الماي"  # "The meter is stopped although we use water"
ids = T.encode(tok, [SENT], cls=False)
TOKENS = [tok.id_to_token(i) for i in ids[0].tolist()]
print(len(TOKENS), "tokens:", TOKENS)

# %% [markdown]
# ## 2 · Attention by hand for one head (7')
#
# One head, in three lines: every token asks a question (its **query**), every token offers a label (its
# **key**), and the match between a query and each key, after a softmax, says how much of each token's
# **value** to take.
#
# `weights = softmax(Q · Kᵀ / √d_k)`; in a decoder, every position to the right of the asking token is set to
# minus infinity before the softmax.
#
# ✏️ **Exercise** · Write `attention_weights(q, k, causal)`. `q` and `k` have shape (tokens, d_k). Return a
# (tokens, tokens) tensor whose row `i` is the attention of token `i` over all tokens.

# %%
def attention_weights(q: torch.Tensor, k: torch.Tensor, causal: bool = False) -> torch.Tensor:
    # BEGIN SOLUTION
    scores = q @ k.T / math.sqrt(q.shape[-1])
    if causal:
        scores = scores.masked_fill(~torch.ones_like(scores, dtype=torch.bool).tril(), float("-inf"))
    return scores.softmax(-1)
    # END SOLUTION


# %%
def head_inputs(model, ids, layer=0, head=0):
    """The queries and keys of one head of one layer, taken from inside the model."""
    with torch.no_grad():
        model(ids)  # fills .last in every attention block
        x = model.tok(ids) + model.pos(torch.arange(ids.shape[1]))[None]
        for blk in model.blocks[:layer]:
            x = blk(x, ids != 0)
        blk = model.blocks[layer]
        h = blk.ln1(x)[0]
        dk = model.cfg.d // model.cfg.heads
        q = blk.attn.q(h).view(-1, model.cfg.heads, dk)[:, head]
        k = blk.attn.k(h).view(-1, model.cfg.heads, dk)[:, head]
    return q, k, blk.attn.last[0, head]


for name, model in [("encoder", enc), ("decoder", dec)]:
    q, k, inside = head_inputs(model, ids)
    try:
        mine = attention_weights(q, k, causal=model.cfg.causal)
    except NotImplementedError:
        mine = inside  # safety net: the model's own weights
    assert torch.allclose(mine, inside, atol=1e-5), f"{name}: your weights differ from the model's"
    assert torch.allclose(mine.sum(-1), torch.ones(len(TOKENS)), atol=1e-5)
    print(f"✅ {name}: layer 0, head 0 matches the model; every row sums to 1")

# %% [markdown]
# ## 3 · Read the maps: who attends to whom (7')
#
# Row = the token that asks, column = the token it looks at. `model.attentions()` returns one tensor per
# layer: (batch, heads, asking token, token looked at).
#
# ✏️ **Exercise** · Write `mass_on_later_tokens(att)`: for a (tokens, tokens) map, the share of all attention
# that goes to a token to the right of the asking token (above the diagonal), as a float.

# %%
def mass_on_later_tokens(att: torch.Tensor) -> float:
    # BEGIN SOLUTION
    return float(att.triu(1).sum() / att.sum())
    # END SOLUTION


# %%
def maps(model):
    with torch.no_grad():
        model(ids)
    return [a[0].mean(0) for a in model.attentions()]  # one (tokens, tokens) map per layer, averaged over heads


later = G.attempt(lambda: {name: round(mass_on_later_tokens(maps(m)[0]), 3) for name, m in [("encoder", enc), ("decoder", dec)]}, "d13_later")
print("share of attention on later tokens, layer 0:", later)
assert later["decoder"] == 0 and later["encoder"] > 0.2

for name, model in [("encoder", enc), ("decoder", dec)]:
    a = maps(model)[-1]
    print(f"\n{name}, last layer: the token each token looks at most")
    print(pd.DataFrame({"token": TOKENS, "looks most at": [TOKENS[j] for j in a.argmax(-1).tolist()],
                        "weight": [round(float(w), 2) for w in a.max(-1).values]}).to_string())

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, (name, model) in zip(axes, [("encoder: sees both ways", enc), ("decoder: sees only back", dec)]):
    ax.imshow(maps(model)[0].numpy(), cmap="Blues", vmin=0)
    ax.set(title=name, xlabel="token looked at (position)", ylabel="asking token (position)")
fig.tight_layout()
fig.savefig(G.lab_dir("D1_3") / "attention_maps.png", dpi=120)
plt.show()

# %% [markdown]
# 🔍 The decoder's map is a triangle: nothing above the diagonal. In the encoder, the first word can use the
# last. Attention weights say where a head looked; they are not an explanation of the model's decision.

# %% [markdown]
# ## 4 · Fill the gap, or predict the next token (7')
#
# The encoder was trained to recover a hidden token from both sides; the decoder to predict the next token
# from the left side only. `model.lm_logits(model(ids))` gives, at every position, a score for every entry of
# the vocabulary.
#
# ✏️ **Exercise** · Write `fill_mask(text, k)`: encode `text` (it contains `[MASK]`) for the encoder with
# `T.encode(tok, [text])` and return the `k` most likely tokens at the masked position. Write
# `next_tokens(prefix, k)`: the `k` most likely next tokens after `prefix` from the decoder
# (`T.encode(tok, [prefix], cls=False)`).

# %%
MASK_ID = tok.token_to_id("[MASK]")


def fill_mask(text: str, k: int = 5) -> list[str]:
    # BEGIN SOLUTION
    x = T.encode(tok, [text])
    pos = (x[0] == MASK_ID).nonzero()[0, 0]
    with torch.no_grad():
        logits = enc.lm_logits(enc(x))[0, pos]
    return [tok.id_to_token(i) for i in logits.topk(k).indices.tolist()]
    # END SOLUTION


def next_tokens(prefix: str, k: int = 5) -> list[str]:
    # BEGIN SOLUTION
    x = T.encode(tok, [prefix], cls=False)
    with torch.no_grad():
        logits = dec.lm_logits(dec(x))[0, -1]
    return [tok.id_to_token(i) for i in logits.topk(k).indices.tolist()]
    # END SOLUTION


# %%
pairs = [("الماي [MASK] عندنا من الصبح", "الماي"), ("the water [MASK] is very weak on the second floor", "the water"),
         ("el [MASK] maqtoo3 3indna", "el")]
guesses = G.attempt(lambda: [{"gap": g, "encoder fills": fill_mask(g), "prefix": p, "decoder continues": next_tokens(p)} for g, p in pairs], "d13_guesses")
for row in guesses:
    print(f"{row['gap']}\n   encoder fills the gap with : {row['encoder fills']}\n   decoder after '{row['prefix']}' : {row['decoder continues']}\n")
assert all(len(r["encoder fills"]) == 5 and len(r["decoder continues"]) == 5 for r in guesses)

# %% [markdown]
# 🔍 In the second pair the encoder has read "is very weak on the second floor" before it fills the gap. The
# decoder has only "the water" and must keep every continuation open. For a decision about a whole message
# (which department?), which of the two has the better view?

# %% [markdown]
# ## ✱ Stretch · Change one letter of the sentence and compare the maps
#
# Add a diacritic to العداد. How many tokens does the sentence have now, and what happens to every position
# after the changed word?

# %%
# BEGIN SOLUTION
other = T.encode(tok, [SENT.replace("العداد", "العَداد")], cls=False)
print([tok.id_to_token(i) for i in other[0].tolist()])
print("tokens:", ids.shape[1], "→", other.shape[1], "| one diacritic splits the word into more pieces, so every later position shifts")
# END SOLUTION

# %% [markdown]
# ## Checkpoint
#
# 1. Why must every row of an attention map sum to 1?
# 2. What single line of code turns the encoder's attention into the decoder's?
# 3. You must label 2 million messages a day with one of eight departments. Which family, and why?
#
# <details><summary>Answers</summary>
#
# 1. The softmax turns each token's scores into shares of a whole: the token's new state is a weighted average
#    of the values.
# 2. The mask that sets every score to the right of the asking token to minus infinity before the softmax.
# 3. An encoder with a classification head: it reads the whole message in both directions in one pass and
#    answers with eight scores. A decoder generates the label token by token and is larger for the same quality.
#
# </details>
#
# **Carry forward.** Lab 3.2 puts a classification head on this encoder; Lab 2.1 samples from this decoder.
