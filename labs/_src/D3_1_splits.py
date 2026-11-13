# %% [markdown]
# # Lab 3.1 · A split without leakage
#
# Day 3 · Session 1 (Fine-tuning data) · 25 min. Before anything is fine-tuned, the test score has to be one
# you can believe. Sawaqi's customers write again when nobody answers: the same complaint, the same facts, a
# new greeting. If one copy is in the training set and the other in the test set, the model is tested on
# messages it has seen. You find those near-duplicates, measure how much a random split inflates the score,
# check the labels by variety, and write the split and its data card.
#
# **Reference state.** The 1,100 labelled messages of `data/messages.csv` (`G.labelled()`), each with its
# `customer_id`. Nothing from another lab: the baseline model is rebuilt here by `G.baseline()`.
#
# **Tasks**
# - [ ] 1 · Find near-duplicate messages per customer (5')
# - [ ] 2 · Random split against customer split (8')
# - [ ] 3 · Label balance by dialect (6')
# - [ ] 4 · Write the split and a data card (6')
# - [ ] ✱ Stretch · A split that is grouped by customer and stratified by department

# %%
#@setup
G.banner("Lab 3.1")

# %%
import numpy as np
import pandas as pd

DATA = G.labelled().drop(columns=["split"])
print(len(DATA), "labelled messages from", DATA["customer_id"].nunique(), "customers")
print(DATA["customer_id"].value_counts().describe()[["mean", "max"]].round(2).to_dict())

# %% [markdown]
# ## 1 · Find near-duplicate messages per customer (5')
#
# Two messages are near-duplicates here when the same customer wrote both and their normalised word sets
# overlap strongly: Jaccard similarity (shared words / all words) of 0.6 or more.
#
# ✏️ **Exercise** · Write `near_duplicates(df, threshold)`: a list of `(msg_id, msg_id)` pairs. Compare only
# messages of the same customer.

# %%
def words(text: str) -> set:
    return set(G.normalise(text).split())


def near_duplicates(df: pd.DataFrame, threshold: float = 0.6) -> list:
    # BEGIN SOLUTION
    pairs = []
    for _, group in df.groupby("customer_id"):
        rows = list(zip(group["msg_id"], map(words, group["text"])))
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i][1], rows[j][1]
                if len(a & b) / max(1, len(a | b)) >= threshold:
                    pairs.append((rows[i][0], rows[j][0]))
    return pairs
    # END SOLUTION


# %%
PAIRS = [tuple(p) for p in G.attempt(lambda: near_duplicates(DATA), "d31_pairs")]
in_pair = {m for p in PAIRS for m in p}
print(len(PAIRS), "near-duplicate pairs |", len(in_pair), "messages have at least one near-duplicate",
      f"({len(in_pair) / len(DATA):.0%} of the labelled set)")
a, b = PAIRS[0]
print(DATA.set_index("msg_id").loc[[a, b], ["customer_id", "department", "text"]].to_string())
assert len(PAIRS) > 100 and all(DATA.set_index("msg_id").loc[x, "customer_id"] == DATA.set_index("msg_id").loc[y, "customer_id"] for x, y in PAIRS[:50])

# %% [markdown]
# ## 2 · Random split against customer split (8')
#
# Same data, same model, same sizes (400 to train, 400 to test), two ways to split. In the **random** split
# every message is drawn independently. In the **customer** split every customer is wholly on one side.
#
# ✏️ **Exercise** · Write `random_split(df, seed)` and `customer_split(df, seed)`. Both return `(train, test)`
# with about 400 messages each. For the customer split, shuffle the customer ids and fill the test set
# customer by customer until it holds 400 messages, then the training set the same way.
# Then write `leaked_share(train, test)`: the share of test messages with a near-duplicate (from `PAIRS`) in
# the training set.

# %%
def random_split(df: pd.DataFrame, seed: int):
    # BEGIN SOLUTION
    shuffled = df.sample(frac=1, random_state=seed)
    return shuffled.iloc[400:800], shuffled.iloc[:400]
    # END SOLUTION


def customer_split(df: pd.DataFrame, seed: int):
    # BEGIN SOLUTION
    ids = df["customer_id"].drop_duplicates().sample(frac=1, random_state=seed)
    sizes = df["customer_id"].value_counts()
    side, n_test, n_train = {}, 0, 0
    for cid in ids:
        if n_test < 400:
            side[cid], n_test = "test", n_test + sizes[cid]
        elif n_train < 400:
            side[cid], n_train = "train", n_train + sizes[cid]
    where = df["customer_id"].map(side)
    return df[where == "train"], df[where == "test"]
    # END SOLUTION


def leaked_share(train: pd.DataFrame, test: pd.DataFrame) -> float:
    # BEGIN SOLUTION
    tr, te = set(train["msg_id"]), set(test["msg_id"])
    leaked = {y for x, y in PAIRS if x in tr and y in te} | {x for x, y in PAIRS if y in tr and x in te}
    return len(leaked) / len(test)
    # END SOLUTION


# %%
def compare_splits(seeds=(1, 2, 3, 4, 5)) -> list[dict]:
    rows = []
    for name, splitter in [("random", random_split), ("by customer", customer_split)]:
        for seed in seeds[: G.pick(5, 2)]:
            train, test = splitter(DATA, seed)
            pred = G.baseline().fit(train["text"], train["department"]).predict(test["text"])
            f1 = G.by_variety(test["department"], pred, test["variety"]).iloc[0]["macro_f1"]
            rows.append({"split": name, "seed": seed, "train": len(train), "test": len(test),
                         "leaked_share": round(leaked_share(train, test), 3), "macro_f1": float(f1)})
    return rows


runs = pd.DataFrame(G.attempt(compare_splits, "d31_splits"))
summary = runs.groupby("split", sort=False)[["leaked_share", "macro_f1"]].mean().round(3)
print(runs.to_string(index=False))
print()
print(summary.to_string())
gap = summary.loc["random", "macro_f1"] - summary.loc["by customer", "macro_f1"]
print(f"\nthe random split reports {gap:+.3f} macro F1 that a new customer will not see")
assert summary.loc["by customer", "leaked_share"] == 0 and summary.loc["random", "leaked_share"] > 0.05 and gap > 0

# %% [markdown]
# 🔍 The model did not get better between the two rows; the test got easier. Which of the two numbers predicts
# what happens when a customer who never wrote before sends a message?

# %% [markdown]
# ## 3 · Label balance by dialect (6')
#
# A model learns a department in a variety from the examples it has of that pair. Count them.
#
# ✏️ **Exercise** · Write `thin_cells(train, minimum)`: the list of `(department, variety)` pairs with fewer
# than `minimum` training messages, pairs with zero included.

# %%
def thin_cells(train: pd.DataFrame, minimum: int = 5) -> list:
    # BEGIN SOLUTION
    counts = pd.crosstab(train["department"], train["variety"]).reindex(index=G.DEPARTMENTS, columns=G.VARIETIES, fill_value=0)
    return [(d, v) for d in G.DEPARTMENTS for v in G.VARIETIES if counts.loc[d, v] < minimum]
    # END SOLUTION


# %%
try:
    TRAIN, TEST = customer_split(DATA, 1)
except NotImplementedError:
    TRAIN, TEST = G.messages("train"), G.messages("test")
print(pd.crosstab(TRAIN["department"], TRAIN["variety"]).reindex(index=G.DEPARTMENTS, columns=G.VARIETIES, fill_value=0).to_string())
thin = [tuple(c) for c in G.attempt(lambda: thin_cells(TRAIN), "d31_thin")]
print("\nfewer than 5 training messages:", thin)
assert any(v == "arabizi" for _, v in thin)

# %% [markdown]
# 🔍 Every thin cell is a promise the model cannot keep. Three honest options: label more messages of that
# kind, report the cell as not supported in the model card, or send those messages to a person. Which thin
# cells did Lab 1.4's baseline get wrong?

# %% [markdown]
# ## 4 · Write the split and a data card (6')
#
# ✏️ **Exercise** · Write `overlap(train, test)`: the number of customers that appear on both sides. Then run
# the cell: it writes the split and a data card with the numbers of this lab.

# %%
def overlap(train: pd.DataFrame, test: pd.DataFrame) -> int:
    # BEGIN SOLUTION
    return len(set(train["customer_id"]) & set(test["customer_id"]))
    # END SOLUTION


# %%
try:
    shared = overlap(TRAIN, TEST)
except NotImplementedError:
    shared = len(set(TRAIN["customer_id"]) & set(TEST["customer_id"]))
assert shared == 0, "a customer is on both sides"
out = G.lab_dir("D3_1")
split = pd.concat([TRAIN.assign(split="train"), TEST.assign(split="test")])[["msg_id", "customer_id", "split"]]
split.to_csv(out / "split.csv", index=False)
card = "\n".join([
    "# Data card: Sawaqi routing set", "",
    f"- Messages: {len(TRAIN)} train, {len(TEST)} test, from {DATA['customer_id'].nunique()} customers. Synthetic, generated with a fixed seed.",
    f"- Split: by customer. Customers on both sides: {shared}.",
    f"- Near-duplicates: {len(PAIRS)} pairs within customers; a random split leaks {summary.loc['random', 'leaked_share']:.0%} of the test set"
    f" and reports {gap:+.3f} macro F1.",
    f"- Varieties in train: {TRAIN['variety'].value_counts().to_dict()}",
    f"- Department and variety pairs with fewer than 5 training messages: {thin}",
    "- Personal data: names, phone numbers and account numbers are synthetic; they are masked before text leaves Sawaqi (rule R-10).",
    "- Not for: any decision about a real customer.", ""])
(out / "DATA_CARD.md").write_text(card, encoding="utf-8")
print(card)

# %% [markdown]
# ## ✱ Stretch · A split that is grouped by customer and stratified by department
#
# `sklearn.model_selection.StratifiedGroupKFold` keeps groups together and keeps the label mix similar in
# every fold. Use one of four folds as the test set and compare the department mix of both sides.

# %%
# BEGIN SOLUTION
from sklearn.model_selection import StratifiedGroupKFold

tr_idx, te_idx = next(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=1).split(DATA, DATA["department"], DATA["customer_id"]))
mix = pd.DataFrame({"train": DATA.iloc[tr_idx]["department"].value_counts(normalize=True),
                    "test": DATA.iloc[te_idx]["department"].value_counts(normalize=True)}).round(3)
print(mix.to_string())
# END SOLUTION

# %% [markdown]
# ## Checkpoint
#
# 1. A colleague tunes six learning rates and reports the best test score. What is wrong, and what is the fix?
# 2. Why split by customer and not by message text (dropping exact duplicates)?
# 3. When is fine-tuning worth it at all, given Day 2's prompted model?
#
# <details><summary>Answers</summary>
#
# 1. The test set chose the learning rate, so it is no longer unseen: the reported score is the best of six
#    draws. Choose on the validation split; touch the test split once, at the end.
# 2. The copies are not exact: a new greeting, a typo. And the customer is what repeats in production, with the
#    same account, district and way of writing.
# 3. When you have labels, the task is narrow and stable, and volume or latency make a hosted model costly:
#    a small encoder answers in milliseconds on a processor. When the task changes every month, prompt.
#
# </details>
#
# **Carry forward.** `artifacts/D3_1/split.csv` and `DATA_CARD.md`. Labs 3.2 and 3.3 use the supplied split in
# `data/split.csv`, which was made the same way and also holds a validation set.
