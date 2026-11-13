# %% [markdown]
# # Lab 2.3 · Grounded replies from the rule book
#
# Day 2 · Session 3 (Retrieval-augmented generation) · 25 min. Customers ask what a cubic metre costs, when a
# bill is due, what a meter test costs. The answers are in the tariff rule book that Latifa maintains, and
# nowhere in a model's weights. You index the rule book, measure retrieval, generate replies that cite the
# rule they rest on, and check that Ghayth refuses when the rules are silent.
#
# **Reference state.** `data/rules/R-01 … R-10` (each rule in English and Arabic) and
# `data/rule_questions.jsonl`: 16 questions in three varieties, 12 with the rule that answers them and a key
# fact, 4 that the rule book does not answer. Nothing from another lab. Replies come from a hosted model
# through `G.chat` (16 calls, cached); without a working key a scripted stand-in answers.
#
# **Tasks**
# - [ ] 1 · Chunk and index the rule book (6')
# - [ ] 2 · Retrieval: hit at 3 on 16 questions (5')
# - [ ] 3 · Grounded replies with citations (8')
# - [ ] 4 · Score citations and refusals (6')
# - [ ] ✱ Stretch · Ask in Gulf dialect and in Arabizi: does retrieval still find the rule?

# %%
#@setup
G.banner("Lab 2.3")

# %%
import math
import re
from collections import Counter

import pandas as pd

RULES = G.rules()
QUESTIONS = G.jsonl("rule_questions.jsonl")
ANSWERABLE = [q for q in QUESTIONS if q["rule"]]
print(len(RULES), "rules |", len(QUESTIONS), "questions, of which", len(QUESTIONS) - len(ANSWERABLE), "have no answer in the rules")
print(RULES["R-06"][:260])

# %% [markdown]
# ## 1 · Chunk and index the rule book (6')
#
# Every rule file holds an English part and, after a `## ` heading, an Arabic part. A chunk is one passage
# small enough to quote and large enough to answer: here, one language of one rule. Keep the rule id with
# every chunk: it is what the reply will cite.
#
# ✏️ **Exercise** · Write `chunk_rules(rules)`: a list of dictionaries with `rule` (`R-06`), `lang` (`en` or
# `ar`) and `text`. The English part is everything before the line that starts with `## `; drop the `#`
# marks of the headings and keep the heading words.

# %%
def chunk_rules(rules: dict) -> list[dict]:
    # BEGIN SOLUTION
    chunks = []
    for rid, body in rules.items():
        en, ar = body.split("\n## ", 1)
        chunks.append({"rule": rid, "lang": "en", "text": en.lstrip("# ").strip()})
        chunks.append({"rule": rid, "lang": "ar", "text": ar.strip()})
    return chunks
    # END SOLUTION


# %%
try:
    CHUNKS = chunk_rules(RULES)
except NotImplementedError:
    CHUNKS = [{"rule": rid, "lang": lang, "text": part.lstrip("# ").strip()} for rid, body in RULES.items()
              for lang, part in zip(("en", "ar"), body.split("\n## ", 1))]
assert len(CHUNKS) == 20 and {c["lang"] for c in CHUNKS} == {"en", "ar"}
assert all("##" not in c["text"] and c["text"] for c in CHUNKS)
assert any("؀" <= ch <= "ۿ" for ch in CHUNKS[1]["text"]) and not any("؀" <= ch <= "ۿ" for ch in CHUNKS[0]["text"])


def tokens(text: str) -> list[str]:
    """Normalised words, cut to six letters so that a word and its suffixed forms meet."""
    return [w[:6] for w in re.findall(r"\w+", G.normalise(text)) if len(w) > 1]


DOCS = [tokens(c["text"]) for c in CHUNKS]
DF = Counter(w for d in DOCS for w in set(d))
AVG = sum(map(len, DOCS)) / len(DOCS)


def bm25_search(query: str, k: int = 3) -> list[dict]:
    """BM25: a rare word counts more, a repeated word counts less each time, a long chunk is not favoured."""
    scores = []
    for doc in DOCS:
        tf, s = Counter(doc), 0.0
        for w in tokens(query):
            if w in tf:
                idf = math.log(1 + (len(DOCS) - DF[w] + 0.5) / (DF[w] + 0.5))
                s += idf * tf[w] * 2.5 / (tf[w] + 1.5 * (0.25 + 0.75 * len(doc) / AVG))
        scores.append(s)
    order = sorted(range(len(DOCS)), key=lambda i: -scores[i])[:k]
    return [CHUNKS[i] for i in order if scores[i] > 0]


print(len(CHUNKS), "chunks |", [c["rule"] + "/" + c["lang"] for c in bm25_search("How much does a meter test cost?")])

# %% [markdown]
# ## 2 · Retrieval: hit at 3 on 16 questions (5')
#
# Measure the retriever alone, before any model writes a word. **Hit at 3**: the rule that answers the
# question is among the top three chunks. Only the 12 answerable questions have a gold rule.
#
# ✏️ **Exercise** · Write `hit_at_3(search, questions)`: a dictionary with the share of hits for each variety
# of the questions and for `all`, rounded to 2 decimals.

# %%
def hit_at_3(search, questions: list[dict]) -> dict:
    # BEGIN SOLUTION
    per = {}
    for q in questions:
        per.setdefault(q["variety"], []).append(q["rule"] in [c["rule"] for c in search(q["question"], 3)])
    out = {v: round(sum(h) / len(h), 2) for v, h in per.items()}
    out["all"] = round(sum(sum(h) for h in per.values()) / len(questions), 2)
    return out
    # END SOLUTION


# %%
hits = G.attempt(lambda: hit_at_3(bm25_search, ANSWERABLE), "d23_hits")
print("hit at 3:", hits)
for q in ANSWERABLE:
    got = [c["rule"] for c in bm25_search(q["question"], 3)]
    if q["rule"] not in got:
        print("  miss:", q["qid"], q["variety"], q["question"], "→", got, "| gold", q["rule"])
assert hits["all"] >= 0.75

# %% [markdown]
# 🔍 Read each miss. Is the rule's wording different from the customer's (a dialect word, a synonym)? A miss
# here cannot be repaired by the model: it never sees the rule.

# %% [markdown]
# ## 3 · Grounded replies with citations (8')
#
# The prompt gives the model the retrieved chunks, each with its rule id, and three instructions: answer from
# the rules only, cite the rule id in square brackets, and reply exactly `NOT_IN_RULES` when the rules do not
# answer.
#
# ✏️ **Exercise** · Write `answer(question)`: retrieve the top three chunks, build the user message in the
# format below, call `G.chat` with `SYSTEM` (`max_tokens=200`) and return a dictionary with `reply` and
# `retrieved` (the list of rule ids).
#
# ```
# RULES
# [R-02] <text of the chunk>
# [R-04] <text of the chunk>
#
# QUESTION: <the question>
# ```

# %%
SYSTEM = ("You answer customers of Sawaqi Water Company from the RULES given in the message and from nothing else. "
          "Answer in the language of the question, in one or two sentences, and cite the rule you used in square "
          "brackets, like [R-02]. If the RULES do not answer the question, reply exactly: NOT_IN_RULES")


def answer(question: str) -> dict:
    # BEGIN SOLUTION
    chunks = bm25_search(question, 3)
    context = "\n".join(f"[{c['rule']}] {c['text']}" for c in chunks)
    r = G.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"RULES\n{context}\n\nQUESTION: {question}"}],
               max_tokens=200)
    return {"reply": r["text"].strip(), "retrieved": [c["rule"] for c in chunks]}
    # END SOLUTION


# %%
answers = G.attempt(lambda: [dict(q, **answer(q["question"])) for q in QUESTIONS], "d23_answers")
for a in answers[:3] + answers[-2:]:
    print(f"{a['qid']} {a['question']}\n    → {a['reply'][:200]}\n")
assert len(answers) == 16 and all(a["reply"] for a in answers)

# %% [markdown]
# ## 4 · Score citations and refusals (6')
#
# Four numbers, each about a different part:
#
# | Number | Over | Part it measures |
# |---|---|---|
# | `cited_right` | 12 answerable | the reply cites the gold rule |
# | `has_key_fact` | 12 answerable | the reply contains the key fact (a price, a number of days) |
# | `false_refusals` | 12 answerable | the reply is `NOT_IN_RULES` although a rule answers |
# | `refused` | 4 silent | the reply is `NOT_IN_RULES` |
#
# ✏️ **Exercise** · Write `score_answers(answers)`: a dictionary with these four counts. Citations are every
# `R-` and two digits inside square brackets in the reply.

# %%
def score_answers(answers: list[dict]) -> dict:
    # BEGIN SOLUTION
    out = {"cited_right": 0, "has_key_fact": 0, "false_refusals": 0, "refused": 0}
    for a in answers:
        refused = a["reply"].strip() == "NOT_IN_RULES"
        if a["rule"] is None:
            out["refused"] += refused
        else:
            out["cited_right"] += a["rule"] in re.findall(r"\[(R-\d\d)\]", a["reply"])
            out["has_key_fact"] += a["key_fact"] in a["reply"]
            out["false_refusals"] += refused
    return out
    # END SOLUTION


# %%
scores = G.attempt(lambda: score_answers(answers), "d23_scores")
print(scores, "|", "stand-in (scripted, not a model)" if G.standin() else G.CHAT_MODEL)
rows = pd.DataFrame([{"qid": a["qid"], "variety": a["variety"], "gold": a["rule"], "retrieved": " ".join(a["retrieved"]),
                      "gold_retrieved": a["rule"] in a["retrieved"] if a["rule"] else None, "reply": a["reply"][:60]} for a in answers])
print(rows.to_string(index=False))
out = G.lab_dir("D2_3")
rows.to_csv(out / "grounded_answers.csv", index=False)
assert set(scores) == {"cited_right", "has_key_fact", "false_refusals", "refused"} and scores["cited_right"] <= 12 and scores["refused"] <= 4

# %% [markdown]
# 🔍 For every answerable question that went wrong, decide where: the gold rule was **not retrieved** (fix the
# retriever), or it was retrieved and the reply is still wrong (fix the prompt or the model). For every silent
# question that was answered: which retrieved rule did the reply lean on, and why did it look relevant?

# %% [markdown]
# ## ✱ Stretch · Ask in Gulf dialect and in Arabizi: does retrieval still find the rule?
#
# The rule book is in MSA and English. Ask the same thing three ways and compare the top three rules. What
# would you add to the index, or do to the question, so that the Arabizi form finds R-07?

# %%
# BEGIN SOLUTION
for q in ["كم تكلفة فحص العداد؟", "بكم تفحصون العداد؟", "bkam tif7soon el 3adad?"]:
    print(q, "→", [c["rule"] for c in bm25_search(q, 3)])
# END SOLUTION

# %% [markdown]
# ## Checkpoint
#
# 1. The right rule was retrieved and the reply states the wrong price. Where do you look?
# 2. Why does the refusal have to be an exact string?
# 3. The rule book changes every quarter. What has to be rebuilt, and what stays?
#
# <details><summary>Answers</summary>
#
# 1. At the generator: the prompt (does it say to copy figures from the rules?), the chunk as the model saw it
#    (two tiers in one sentence), and the sampling settings. The retriever did its job.
# 2. So that code can act on it: route the question to a person, count refusals, test them. A polite sentence
#    that means "I do not know" cannot be counted reliably.
# 3. The index is rebuilt from the new files, in minutes, and the evaluation questions are checked against the
#    new rules. The model, the prompt and the code stay.
#
# </details>
#
# **Carry forward.** `artifacts/D2_3/grounded_answers.csv`. Retrieval and agents in depth are another course
# (AE-304); here the rule book is one of Ghayth's sources.
