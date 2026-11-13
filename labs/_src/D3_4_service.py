# %% [markdown]
# # Lab 3.4 · A small service with a model card
#
# Day 3 · Session 4 (Evaluate and deploy) · 25 min. A model in a notebook routes nothing. Ibrahim needs one
# address that takes a message and answers with a department, the extracted fields and, when the model
# should not decide, a hand-off to a person. You wrap the chosen model in a small API with a demo page, pass
# the smoke test, measure latency on this processor and draft the model card with results by variety.
#
# **Reference state.** Two routing models that need nothing from earlier labs: the baseline (trained here in
# a second by `G.baseline()`) and the supplied fine-tuned encoder (`models/ghayth-enc-routing-full.pt`).
# `data/smoke_test.jsonl`: 20 test messages, five per variety, and four edge cases that must go to a person.
# The comparison rows of the other labs come from your own `artifacts/` or from `data/reference/`.
#
# **Tasks**
# - [ ] 1 · An API that routes, extracts and hands off (7')
# - [ ] 2 · Pass the smoke test (5')
# - [ ] 3 · CPU latency, median and 95th percentile (5')
# - [ ] 4 · Draft the model card by variety (8')
# - [ ] ✱ Stretch · Serve the same model with a Gradio or Streamlit demo

# %%
#@setup
G.banner("Lab 3.4")

# %%
import json
import re
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

import tinytf as T

train, test = G.messages("train"), G.messages("test")
CHOSEN = "baseline"  # "baseline" or "encoder" (the supplied fine-tuned encoder): the one your comparison defends


def load_scorer(kind: str):
    """Returns a function from a text to {department: probability}."""
    if kind == "baseline":
        model = G.baseline().fit(train["text"], train["department"])
        return lambda text: dict(zip(model.classes_, model.predict_proba([text])[0].tolist()))
    model, labels = T.load_classifier(G.MODELS / "ghayth-enc-routing-full.pt")
    tok = T.load_tokenizer()

    def score(text: str) -> dict:
        with torch.no_grad():
            return dict(zip(labels, model(T.encode(tok, [text])).softmax(-1)[0].tolist()))
    return score


SCORE = load_scorer(CHOSEN)
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def variety_of(text: str) -> str:
    """A rough guess from the script: 'arabic' (MSA or Gulf), 'arabizi' or 'en'."""
    letters = [ch for ch in text if ch.isalpha()]
    if letters and sum("؀" <= ch <= "ۿ" for ch in letters) / len(letters) > 0.5:
        return "arabic"
    hints = re.findall(r"\b\w*[a-z][235679]\w*|\b[235679][a-z]\w*|\b(?:el|al|fi|min|mai|may|abi|wesh)\b", text.lower())
    return "arabizi" if len(hints) >= 2 else "en"


def fields(text: str) -> dict:
    """Account number and district without a hosted model: exact patterns, as in Lab 2.2's baseline, fixed."""
    plain = text.translate(DIGITS)
    account = re.search(r"(?<!\d)30\d{8}(?!\d)", plain)
    district = next((key for key, names in G.district_names().items() if any(G.normalise(n) in G.normalise(plain) for n in names)), None)
    return {"account_number": account.group(0) if account else None, "district": district}


probs = SCORE("الماي مقطوع عندنا من الصبح")
print(CHOSEN, "|", max(probs, key=probs.get), round(max(probs.values()), 3), "|", variety_of("el mai maqtoo3 3indna"), "|", fields("حسابي ٣٠٦١٠٧٤٦١٠ في النسيم"))

# %% [markdown]
# ## 1 · An API that routes, extracts and hands off (7')
#
# The decision is a function; the API is a thin layer over it. The **hand-off rule** is part of the model: it
# says when Ghayth must not decide. Start with this one:
#
# - fewer than two words with letters: hand off, reason `no request`;
# - the best probability is below the threshold of the message's variety (`THRESHOLDS`): reason `low confidence`;
# - the second-best probability is at least 60% of the best: reason `two departments`.
#
# ✏️ **Exercise** · Write `decide(text)`: a dictionary with `department` (the best one, or `None` when there
# is no request), `confidence` (rounded to 3), `variety`, `handoff`, `reason` (`None` when it is not handed
# off), `account_number` and `district`.

# %%
THRESHOLDS = {"arabic": 0.5, "en": 0.5, "arabizi": 0.7}


def decide(text: str) -> dict:
    # BEGIN SOLUTION
    variety = variety_of(text)
    if len(re.findall(r"[^\W\d_]+", text)) < 2:
        return {"department": None, "confidence": 0.0, "variety": variety, "handoff": True, "reason": "no request",
                "account_number": None, "district": None}
    ranked = sorted(SCORE(text).items(), key=lambda kv: -kv[1])
    (best, p1), (_, p2) = ranked[0], ranked[1]
    reason = "low confidence" if p1 < THRESHOLDS[variety] else "two departments" if p2 >= 0.6 * p1 else None
    return {"department": best, "confidence": round(p1, 3), "variety": variety, "handoff": reason is not None, "reason": reason,
            **fields(text)}
    # END SOLUTION


# %%
def reference_decide(text):
    variety = variety_of(text)
    if len(re.findall(r"[^\W\d_]+", text)) < 2:
        return {"department": None, "confidence": 0.0, "variety": variety, "handoff": True, "reason": "no request", "account_number": None, "district": None}
    ranked = sorted(SCORE(text).items(), key=lambda kv: -kv[1])
    (best, p1), (_, p2) = ranked[0], ranked[1]
    reason = "low confidence" if p1 < THRESHOLDS[variety] else "two departments" if p2 >= 0.6 * p1 else None
    return {"department": best, "confidence": round(p1, 3), "variety": variety, "handoff": reason is not None, "reason": reason, **fields(text)}


try:
    decide("hello")
except NotImplementedError:
    decide = reference_decide


class Message(BaseModel):
    text: str = Field(max_length=2000)


class Decision(BaseModel):
    department: Optional[str]
    confidence: float
    variety: str
    handoff: bool
    reason: Optional[str]
    account_number: Optional[str]
    district: Optional[str]


DEMO = """<!doctype html><meta charset="utf-8"><title>Ghayth</title>
<h1>Ghayth · message routing</h1>
<textarea id="t" rows="4" cols="60" dir="auto" placeholder="Paste a customer message"></textarea><br>
<button onclick="go()">Route</button><pre id="o"></pre>
<script>async function go(){const r=await fetch('route',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({text:document.getElementById('t').value})});
document.getElementById('o').textContent=JSON.stringify(await r.json(),null,2)}</script>"""

app = FastAPI(title="Ghayth", version="0.1")


@app.get("/", response_class=HTMLResponse)
def demo() -> str:
    return DEMO


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": CHOSEN}


@app.post("/route", response_model=Decision)
def route(message: Message) -> Decision:
    return Decision(**decide(message.text))


client = TestClient(app)  # calls the app in this process: no port, no server to stop
print(client.get("/health").json())
print(client.post("/route", json={"text": "السلام عليكم الماي مقطوع عندنا في النسيم من الصبح رقم حسابي 3061074610"}).json())
assert client.get("/").status_code == 200 and "Ghayth" in client.get("/").text
assert client.post("/route", json={"text": "x" * 3000}).status_code == 422, "the schema refuses a message that is too long"

# %% [markdown]
# 🔍 To run it as a real server: put the cells above in `service.py` and start it with
# `uv run uvicorn service:app --port 8000`; the demo page is then at `http://127.0.0.1:8000/` and the
# generated API documentation at `/docs`. `capstone/service.py` is that file.

# %% [markdown]
# ## 2 · Pass the smoke test (5')
#
# A smoke test is small and runs on every change. It does not measure quality; it proves the service is
# alive and behaves at the edges. Here: every call answers 200; each of the four edge cases (an empty
# message, punctuation only, a greeting, three problems at once) is handed off; every real message gets a
# department from the list.
#
# ✏️ **Exercise** · Write `smoke_test(client, cases)`: the list of failures, each a dictionary with the case's
# `id` and `why` (`status`, `edge case not handed off` or `no department`). An empty list is a pass.

# %%
CASES = G.jsonl("smoke_test.jsonl")


def smoke_test(client, cases: list[dict]) -> list[dict]:
    # BEGIN SOLUTION
    failures = []
    for case in cases:
        r = client.post("/route", json={"text": case["text"]})
        if r.status_code != 200:
            failures.append({"id": case["id"], "why": "status"})
        elif case["handoff"] and not r.json()["handoff"]:
            failures.append({"id": case["id"], "why": "edge case not handed off"})
        elif not case["handoff"] and r.json()["department"] not in G.DEPARTMENTS:
            failures.append({"id": case["id"], "why": "no department"})
    return failures
    # END SOLUTION


# %%
failures = G.attempt(lambda: smoke_test(client, CASES), "d34_smoke")
print(len(CASES), "cases |", "PASS" if not failures else f"FAIL: {failures}")
answers = [client.post("/route", json={"text": c["text"]}).json() for c in CASES]
real = [(c, a) for c, a in zip(CASES, answers) if not c["handoff"]]
print("of the 20 real messages:", sum(a["department"] == c["department"] for c, a in real), "right department,",
      sum(a["handoff"] for _, a in real), "handed off")
for c, a in zip(CASES[-4:], answers[-4:]):
    print(f"  {c['id']} ({c['note']}): handoff={a['handoff']} reason={a['reason']}")
assert failures == [], "the smoke test must pass before the service is measured"

# %% [markdown]
# ## 3 · CPU latency, median and 95th percentile (5')
#
# ✏️ **Exercise** · Write `measure(client, texts)`: call `/route` once per text, time each call with
# `time.perf_counter()`, and return the list of seconds.

# %%
def measure(client, texts) -> list[float]:
    # BEGIN SOLUTION
    seconds = []
    for t in texts:
        t0 = time.perf_counter()
        client.post("/route", json={"text": t})
        seconds.append(time.perf_counter() - t0)
    return seconds
    # END SOLUTION


# %%
try:
    latency = G.percentiles(measure(client, list(test["text"])[: G.pick(200, 50)]))
    G.keep("service_latency", latency)
except NotImplementedError:
    latency = G.reference("service_latency")
    print("↩️  measure not written yet: using the reference measurement")
print("service latency on this processor, per message:", latency, "| budget: 300 ms at the 95th percentile")
assert latency["p95_ms"] < 300

# %% [markdown]
# ## 4 · Draft the model card by variety (8')
#
# A model card says what the model is for, how it was measured, and where it must not be trusted. The table
# by variety and the hand-off rule are the parts Hessa's team will read.
#
# ✏️ **Exercise** · Write `limits(scores, floor)`: from the table that `G.by_variety` returns for the served
# model, the list of varieties whose macro F1 is below `floor` (leave out the row `all`).

# %%
def limits(scores: pd.DataFrame, floor: float = 0.7) -> list[str]:
    # BEGIN SOLUTION
    return [r["variety"] for _, r in scores.iterrows() if r["variety"] != "all" and r["macro_f1"] < floor]
    # END SOLUTION


# %%
decisions = pd.DataFrame([decide(t) for t in test["text"]])
served = G.by_variety(test["department"], decisions["department"].fillna("none"), test["variety"])
kept = ~decisions["handoff"].to_numpy()
by_var = pd.DataFrame({"handed_off": pd.Series(~kept).groupby(test["variety"].to_numpy()).mean().round(3),
                       "accuracy_when_kept": pd.Series((decisions["department"] == test["department"].to_numpy())[kept]).groupby(test["variety"].to_numpy()[kept]).mean().round(3)})
by_var.loc["all"] = [round(float((~kept).mean()), 3), round(float((decisions["department"] == test["department"].to_numpy())[kept].mean()), 3)]
served = served.merge(by_var, left_on="variety", right_index=True, how="left")
weak = [str(v) for v in G.attempt(lambda: limits(served), "d34_limits")]
print(served.to_string(index=False))
print("below 0.7 macro F1:", weak)

full, lora, base = G.reference("full_ft_log"), G.reference("lora_log"), pd.read_csv(G.supplied("baseline_predictions.csv", "D1_4"))
prompted = pd.read_csv(G.supplied("comparison.csv", "D2_4"), index_col=0) if (G.ARTIFACTS / "D2_4" / "comparison.csv").exists() else pd.DataFrame(G.reference("d24_table")).set_index("index")
rows = [{"approach": "baseline: word and character counts, linear", **G.by_variety(base.y_true, base.y_pred, base.variety).set_index("variety")["macro_f1"].to_dict(),
         "test set": f"{len(base)} messages"},
        {"approach": "prompted model (" + str(prompted.loc["larger", "priced_as"]) + ")", **{k: prompted.loc["larger", k] for k in G.VARIETIES},
         "all": prompted.loc["larger", "macro_f1"], "test set": "40 messages"},
        {"approach": "fine-tuned encoder, every weight", **{r["variety"]: r["macro_f1"] for r in full["test"]}, "test set": f"{full['test'][0]['n']} messages"},
        {"approach": "fine-tuned encoder, low-rank adapter", **{r["variety"]: r["macro_f1"] for r in lora["test"]}, "test set": f"{lora['test'][0]['n']} messages"}]
comparison = pd.DataFrame(rows)[["approach", "all"] + G.VARIETIES + ["test set"]].rename(columns={"all": "macro_f1"})
print()
print(comparison.to_string(index=False))

out = G.lab_dir("D3_4")
card = "\n".join([
    "# Model card: Ghayth routing, version 0.1", "",
    "## What it is for",
    "Routes one customer message of Sawaqi Water Company to one of eight departments and extracts the account number and the district."
    " Messages in Modern Standard Arabic, Gulf dialect, Arabic in Latin script and English. Not for any decision about a customer's bill or service.", "",
    f"## Served model\n{CHOSEN}. Trained on about 400 labelled synthetic messages, split by customer.", "",
    "## Results on the held-out test split, as served (hand-off rule applied)", "```", served.to_string(index=False), "```", "",
    "## The approaches compared (macro F1)", "```", comparison.to_string(index=False), "```",
    "The prompted row comes from 40 messages; replace it with your own Lab 2.4 run. Cost and latency per approach are in that lab's table.", "",
    f"## Latency\n{latency['p50_ms']} ms median, {latency['p95_ms']} ms at the 95th percentile per message, on one laptop processor.", "",
    "## Limits by language variety",
    f"Macro F1 below 0.7 for: {', '.join(weak) or 'none'}. Departments with fewer than 5 training messages in a variety are not supported in that variety (see the data card).", "",
    "## Hand-off rule",
    f"A message goes to a person when it has fewer than two words, when the best probability is below {THRESHOLDS},"
    " or when the second department scores at least 60% of the first.", "",
    "## Data and privacy",
    "All messages are synthetic. Names, phone numbers and account numbers are masked before text reaches a hosted model (rule R-10).", ""])
(out / "MODEL_CARD.md").write_text(card, encoding="utf-8")
comparison.to_csv(out / "comparison.csv", index=False)
G.keep("d34_served", served.to_dict(orient="records"))
G.keep("d34_comparison", comparison.to_dict(orient="records"))
print("\nwrote", (out / "MODEL_CARD.md").relative_to(G.LABS))
assert "arabizi" in card and len(comparison) == 4

# %% [markdown]
# 🔍 Read the column `handed_off` before anything else: the starting rule is cautious. Is that share of
# messages what Hessa's team can take? The thresholds are yours to set, on the validation split.
#
# Open `artifacts/D3_4/MODEL_CARD.md` and finish it in your own words: one sentence on who should not rely
# on this model, and one on what you would measure next. Then change `CHOSEN` at the top, run again and
# compare the two cards. Which one would you sign?

# %% [markdown]
# ## ✱ Stretch · Serve the same model with a Gradio or Streamlit demo
#
# A demo for colleagues who will not call an API. With Gradio, in Colab or after `uv add gradio`:
#
# ```python
# import gradio as gr
# gr.Interface(fn=decide, inputs=gr.Textbox(lines=4, label="Customer message"), outputs=gr.JSON(label="Decision")).launch()
# ```
#
# What does the demo page need to say so that nobody pastes a real customer's message into it?

# %% [markdown]
# ## Checkpoint
#
# 1. The smoke test passes. What do you still not know?
# 2. Why is the hand-off threshold higher for Arabizi?
# 3. What must the model card say about Arabizi, in one sentence a contact-centre lead can act on?
#
# <details><summary>Answers</summary>
#
# 1. Quality. The smoke test proves the service answers and behaves at the edges; the table by variety on the
#    held-out test split says how often it is right.
# 2. The model is right less often on Arabizi at the same confidence, because it saw fewer Arabizi examples.
#    A higher threshold sends more of those messages to a person.
# 3. For example: "Arabizi messages are routed correctly less often than the others; any Arabizi message
#    below 0.7 confidence goes to a person, which is about this share of them."
#
# </details>
#
# **Carry forward.** `artifacts/D3_4/MODEL_CARD.md` and `comparison.csv` are the skeleton of your capstone
# README. `capstone/service.py` is this service as a file you can run.
