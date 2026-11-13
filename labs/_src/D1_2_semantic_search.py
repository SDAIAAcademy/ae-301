# %% [markdown]
# # Lab 1.2 · Semantic search over 2,000 messages
#
# Day 1 · Session 2 (Embeddings and semantic search) · 25 min. Hessa's agents want to find earlier messages
# about the same problem in the same district, whatever variety the customer wrote in. You build a character
# n-gram baseline, embed the archive with a hosted open multilingual embedding model, search it with a flat
# index, and score both by variety.
#
# **Reference state.** `data/messages.csv` (2,000 messages) and the first 20 queries of `data/queries.jsonl`
# (five per variety), each with the messages that are relevant: the same complaint type in the same district,
# in any variety. Nothing from another lab. Embeddings come from a hosted model through `G.embed` and are
# cached in `cache/`; without a working key a lexical stand-in answers, and the banner says so.
#
# **Tasks**
# - [ ] 1 · Normalise and build a character n-gram baseline (6')
# - [ ] 2 · Embed 2,000 messages with a hosted open model (6')
# - [ ] 3 · A flat inner-product index and one query (5')
# - [ ] 4 · Recall at 10 for 20 queries, by variety (8')
# - [ ] ✱ Stretch · Score all 40 queries and add a hybrid of both rankings

# %%
#@setup
G.banner("Lab 1.2")

# %%
import numpy as np
import pandas as pd

ARCHIVE = G.messages()
IDS = list(ARCHIVE["msg_id"])
ALL_QUERIES = G.jsonl("queries.jsonl")
QUERIES = ALL_QUERIES[:20]
print(len(ARCHIVE), "messages |", len(QUERIES), "queries |", pd.Series([q["variety"] for q in QUERIES]).value_counts().to_dict())
q = QUERIES[2]
print(q["variety"], "|", q["query"], "| relevant:", len(q["relevant"]))
print(ARCHIVE.set_index("msg_id").loc[q["relevant"][:4], ["variety", "text"]].to_string())

# %% [markdown]
# 🔍 The query is in one variety; the relevant messages are in all four. A search that matches letters can only
# find the ones that share letters with the query.

# %% [markdown]
# ## 1 · Normalise and build a character n-gram baseline (6')
#
# The baseline: TF-IDF weights of character 2- to 4-grams of the normalised text. It needs no model, and it
# already matches a word in two spellings.
#
# ✏️ **Exercise** · Fit a `TfidfVectorizer` (`analyzer="char_wb"`, `ngram_range=(2, 4)`, `preprocessor=G.normalise`)
# on the archive, and write `ngram_search(query, k)`: the `msg_id` of the `k` most similar messages, best
# first. TF-IDF rows are unit length, so the dot product is the cosine similarity.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# BEGIN SOLUTION
VEC = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), preprocessor=G.normalise)
NGRAMS = VEC.fit_transform(ARCHIVE["text"])
# END SOLUTION


def ngram_search(query: str, k: int = 10) -> list[str]:
    # BEGIN SOLUTION
    scores = (NGRAMS @ VEC.transform([query]).T).toarray().ravel()
    return [IDS[i] for i in np.argsort(-scores)[:k]]
    # END SOLUTION


# %%
try:
    ngram_search(q["query"])
except (NotImplementedError, NameError):
    VEC = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), preprocessor=G.normalise)
    NGRAMS = VEC.fit_transform(ARCHIVE["text"])

    def ngram_search(query, k=10):
        scores = (NGRAMS @ VEC.transform([query]).T).toarray().ravel()
        return [IDS[i] for i in np.argsort(-scores)[:k]]

hits = ngram_search(q["query"])
assert len(hits) == 10 and len(set(hits)) == 10
print("n-gram hits:", sum(h in q["relevant"] for h in hits), "of 10 are relevant")
print(ARCHIVE.set_index("msg_id").loc[hits[:5], ["variety", "text"]].to_string())

# %% [markdown]
# ## 2 · Embed 2,000 messages with a hosted open model (6')
#
# An embedding model turns a text into a vector so that texts with the same meaning are close. A multilingual
# one puts an Arabic message and its English version close together. `G.embed(texts)` calls the hosted model
# in batches and caches every vector, so a second run is free. `G.EMBED_MODEL` names the model; set
# `ANLP_EMBED_MODEL` in `.env` for another one.
#
# ✏️ **Exercise** · Embed the archive into `E` (one row per message, in the order of `IDS`) with
# `G.embed(texts, batch=100)`. Check that every row has length 1.

# %%
# BEGIN SOLUTION
E = G.embed(list(ARCHIVE["text"]), batch=100)
# END SOLUTION

# %%
try:
    E
except NameError:
    E = G.embed(list(ARCHIVE["text"]), batch=100)
assert E.shape[0] == len(ARCHIVE) and np.allclose(np.linalg.norm(E, axis=1), 1, atol=1e-3)
print("embeddings:", E.shape, "| model:", "stand-in (lexical, not a model)" if G.standin() else G.EMBED_MODEL)

# %% [markdown]
# ## 3 · A flat inner-product index and one query (5')
#
# With unit vectors, cosine similarity is a dot product, and the simplest index is the matrix itself: one
# multiplication scores the query against every message. A vector index library (FAISS `IndexFlatIP`, LanceDB)
# does the same and adds approximate search when the archive has millions of rows. At 2,000 rows, exact search
# takes a millisecond.
#
# ✏️ **Exercise** · Write `vector_search(query, k)`: embed the query, score it against `E`, return the `msg_id`
# of the top `k`.

# %%
def vector_search(query: str, k: int = 10) -> list[str]:
    # BEGIN SOLUTION
    scores = E @ G.embed([query])[0]
    return [IDS[i] for i in np.argsort(-scores)[:k]]
    # END SOLUTION


# %%
try:
    vector_search(q["query"])
except NotImplementedError:
    def vector_search(query, k=10):
        return [IDS[i] for i in np.argsort(-(E @ G.embed([query])[0]))[:k]]

hits = vector_search(q["query"])
print("vector hits:", sum(h in q["relevant"] for h in hits), "of 10 are relevant")
print(ARCHIVE.set_index("msg_id").loc[hits[:5], ["variety", "text"]].to_string())

# %% [markdown]
# ## 4 · Recall at 10 for 20 queries, by variety (8')
#
# Recall at 10: of the relevant messages a top-10 list could hold, the share it does hold. A query with 4
# relevant messages can reach 4 of 4; one with 15 can reach 10 at most, so divide by `min(10, relevant)`.
#
# ✏️ **Exercise** · Write `recall_at_10(search, queries)`: a dictionary with the mean recall at 10 for each
# variety of the queries and for `all`, rounded to 3 decimals.

# %%
def recall_at_10(search, queries: list[dict]) -> dict:
    # BEGIN SOLUTION
    per = {}
    for query in queries:
        got = set(search(query["query"], 10)) & set(query["relevant"])
        per.setdefault(query["variety"], []).append(len(got) / min(10, len(query["relevant"])))
    out = {v: round(float(np.mean(per[v])), 3) for v in G.VARIETIES if v in per}
    out["all"] = round(float(np.mean([r for rs in per.values() for r in rs])), 3)
    return out
    # END SOLUTION


# %%
recall = G.attempt(lambda: {"char n-grams": recall_at_10(ngram_search, QUERIES), "embedding": recall_at_10(vector_search, QUERIES)},
                   "d12_recall")
table = pd.DataFrame(recall).T[G.VARIETIES + ["all"]]
print(table.to_string())
assert set(table.columns) == set(G.VARIETIES + ["all"]) and table.notna().all().all()
if G.standin():
    print("\n⚠️  The 'embedding' row above comes from the lexical stand-in: it is one more n-gram search, not a model.")
(G.lab_dir("D1_2") / "recall_at_10.csv").write_text(table.to_csv(), encoding="utf-8")

# %% [markdown]
# 🔍 This table is the evidence of this session. Read it by column. The n-gram baseline finds messages in the
# query's own script and misses the others: an Arabizi query shares no letters with an Arabic-script message.
# Where does your embedding model close that gap, and where does it not? Then try the query below.

# %%
account = ARCHIVE[ARCHIVE["account_number"].notna()].iloc[5]
for name, search in [("char n-grams", ngram_search), ("embedding", vector_search)]:
    top = ARCHIVE.set_index("msg_id").loc[search(account["account_number"], 5)]
    print(f"{name}: account {account['account_number']} found in", int((top["account_number"] == account["account_number"]).sum()), "of the top 5")

# %% [markdown]
# 🔍 An account number has no meaning to embed. Exact identifiers belong in an exact lookup, beside the vector
# search, not inside it.

# %% [markdown]
# ## ✱ Stretch · Score all 40 queries and add a hybrid of both rankings
#
# Reciprocal rank fusion: a message's score is the sum over both rankings of `1 / (60 + rank)`. Score the
# hybrid on all 40 queries.

# %%
# BEGIN SOLUTION
def hybrid_search(query: str, k: int = 10) -> list[str]:
    score = {}
    for ranking in (ngram_search(query, 50), vector_search(query, 50)):
        for rank, mid in enumerate(ranking):
            score[mid] = score.get(mid, 0) + 1 / (60 + rank)
    return sorted(score, key=score.get, reverse=True)[:k]


print(pd.DataFrame({name: recall_at_10(s, ALL_QUERIES) for name, s in
                    [("char n-grams", ngram_search), ("embedding", vector_search), ("hybrid", hybrid_search)]}).T.to_string())
# END SOLUTION

# %% [markdown]
# ## Checkpoint
#
# 1. Why divide by `min(10, relevant)` and not by the number of relevant messages?
# 2. The archive grows to 20 million messages. What changes in step 3, and what stays?
# 3. Twenty queries, five per variety. How far would you trust a difference of 0.05 between two rows?
#
# <details><summary>Answers</summary>
#
# 1. A list of ten cannot hold fifteen relevant messages; dividing by fifteen would punish a perfect list.
# 2. The exact dot product becomes too slow and too large for memory: an approximate index (FAISS, LanceDB)
#    replaces the matrix. The embeddings, the metric and the evaluation stay the same.
# 3. Not far. With five queries, one query moves a variety's recall by up to 0.2. Score all 40 and look at the
#    queries one by one before claiming a winner.
#
# </details>
#
# **Carry forward.** `artifacts/D1_2/recall_at_10.csv` is your table of recall at 10 by variety. Day 2 searches
# the rule book in the same way.
