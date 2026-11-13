# %% [markdown]
# # Lab 2.4 · Two open-weight models against the baseline
#
# Day 2 · Session 4 (Model selection and evaluation) · 25 min. A prompted model needs no training data: you
# describe the eight departments and ask. Is it better than Day 1's baseline, for which customers, how fast,
# and at what price? You run two open-weight models as zero-shot routers on 40 labelled messages and put them
# in one table with the baseline: macro F1, F1 by variety, latency and cost per thousand messages.
#
# **Reference state.** 40 test messages, ten per variety, drawn with a fixed seed; the baseline's predictions
# for them (your own from Lab 1.4, otherwise `data/reference/baseline_predictions.csv`); the provider's
# published prices (live, otherwise `data/reference/prices.json`). The models are called through `G.chat`
# (80 calls, cached; 20 messages in quick mode); without a working key a scripted stand-in answers.
#
# **Tasks**
# - [ ] 1 · A zero-shot routing prompt and a label parser (6')
# - [ ] 2 · Run both models on 40 messages (7')
# - [ ] 3 · Macro F1 and F1 by variety (6')
# - [ ] 4 · Latency, cost per thousand, decision (6')
# - [ ] ✱ Stretch · A cascade: the baseline first, the model only when it is unsure

# %%
#@setup
G.banner("Lab 2.4")

# %%
import json

import numpy as np
import pandas as pd

test = G.messages("test")
PER_VARIETY = G.pick(10, 5)


def sample_of(per_variety: int) -> pd.DataFrame:
    return pd.concat([test[test["variety"] == v].sample(per_variety, random_state=301) for v in G.VARIETIES]).reset_index(drop=True)


SAMPLE = sample_of(PER_VARIETY)
MODELS = {"larger": G.CHAT_MODEL, "smaller": G.SMALL_MODEL}
print(len(SAMPLE), "messages |", MODELS)

# %% [markdown]
# ## 1 · A zero-shot routing prompt and a label parser (6')
#
# The prompt names every department with one line on what it handles, and asks for the name only. Models do
# not always obey: "Department: leaks", "**Leaks**", "leaks." are all the same answer, and "I think this is
# about a bill" is none.
#
# ✏️ **Exercise** · Write `parse_label(reply)`: the department whose name appears first in the lower-cased
# reply, or `None` when no department name appears.

# %%
SYSTEM = """You route one customer message of a water utility to one department. The message may be in Modern Standard Arabic, Gulf dialect, Arabic in Latin script or English.
Departments:
- billing: a bill that is too high, a charge that appears twice
- payments: a payment that failed, a request to pay in instalments
- leaks: water leaking in the street or at the meter
- supply: no water, weak pressure
- quality: colour, smell or taste of the water
- meters: a meter that is broken or reads wrong, meter readings
- connections: a new connection, moving or closing an account
- wastewater: sewage overflowing or a blocked drain
Reply with the department name only."""


def parse_label(reply: str) -> str | None:
    # BEGIN SOLUTION
    low = reply.lower()
    found = [(low.index(d), d) for d in G.DEPARTMENTS if d in low]
    return min(found)[1] if found else None
    # END SOLUTION


# %%
try:
    parse_label("x")
except NotImplementedError:
    def parse_label(reply):
        found = [(reply.lower().index(d), d) for d in G.DEPARTMENTS if d in reply.lower()]
        return min(found)[1] if found else None

assert parse_label("leaks") == "leaks" and parse_label("Department: **Meters**.") == "meters"
assert parse_label("supply, or maybe leaks") == "supply" and parse_label("I am not sure") is None
print("✅ parse_label passes its four tests")

# %% [markdown]
# ## 2 · Run both models on 40 messages (7')
#
# ✏️ **Exercise** · Write `route(model, text)`: one call to `G.chat` with `SYSTEM` and the user message
# `"Message: " + text`, `model=model`, temperature 0 and `max_tokens=12`. Return a dictionary with `label`
# (parsed), `latency_s`, `prompt_tokens` and `completion_tokens` from the call's result.

# %%
def route(model: str, text: str) -> dict:
    # BEGIN SOLUTION
    r = G.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Message: " + text}],
               model=model, temperature=0.0, max_tokens=12)
    return {"label": parse_label(r["text"]), "latency_s": r["latency_s"], "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"]}
    # END SOLUTION


# %%
runs = G.attempt(lambda: {name: [route(model, t) for t in SAMPLE["text"]] for name, model in MODELS.items()}, "d24_runs")
if any(len(rows) != len(SAMPLE) for rows in runs.values()):  # the reference result covers the full sample of 40
    SAMPLE = sample_of(10)
for name, rows in runs.items():
    unparsed = sum(r["label"] is None for r in rows)
    print(f"{name}: {len(rows)} calls, {unparsed} replies without a department name")
assert all(len(rows) == len(SAMPLE) for rows in runs.values())

# %% [markdown]
# ## 3 · Macro F1 and F1 by variety (6')
#
# The baseline joins the table on the same 40 messages. A reply without a department counts as wrong: label it
# `none`.
#
# ✏️ **Exercise** · Build `PRED`: a dictionary from the name of each approach (`baseline`, `larger`, `smaller`)
# to its list of 40 predicted departments, in the order of `SAMPLE`.

# %%
baseline = pd.read_csv(G.supplied("baseline_predictions.csv", "D1_4")).set_index("msg_id")

# BEGIN SOLUTION
PRED = {"baseline": list(baseline.loc[SAMPLE["msg_id"], "y_pred"])}
for name, rows in runs.items():
    PRED[name] = [r["label"] or "none" for r in rows]
# END SOLUTION

# %%
try:
    PRED
except NameError:
    PRED = {"baseline": list(baseline.loc[SAMPLE["msg_id"], "y_pred"]), **{n: [r["label"] or "none" for r in rows] for n, rows in runs.items()}}
quality = {}
for name, pred in PRED.items():
    t = G.by_variety(SAMPLE["department"], pred, SAMPLE["variety"]).set_index("variety")["macro_f1"]
    quality[name] = t.to_dict()
QUALITY = pd.DataFrame(quality).T.rename(columns={"all": "macro_f1"})
print(QUALITY.to_string())
assert list(QUALITY.index) == ["baseline", "larger", "smaller"]

# %% [markdown]
# 🔍 Ten messages per variety: one message is worth several points of macro F1. Read the Arabizi column, then
# read the messages each model got wrong before you believe a difference.

# %% [markdown]
# ## 4 · Latency, cost per thousand, decision (6')
#
# Cost per thousand messages = 1,000 × (input tokens × input price + output tokens × output price), with the
# mean tokens of your run and the **published** price of the model. A free endpoint costs nothing and has a
# daily limit; for a service you would pay for the same model, so price it at its paid endpoint.
#
# ✏️ **Exercise** · Write `cost_per_1000(rows, price)`: `rows` are the results of `route`; `price` has `prompt`
# and `completion` in USD per token. Round to 4 decimals.

# %%
def cost_per_1000(rows: list[dict], price: dict) -> float:
    # BEGIN SOLUTION
    tin = np.mean([r["prompt_tokens"] or 0 for r in rows])
    tout = np.mean([r["completion_tokens"] or 0 for r in rows])
    return round(float(1000 * (tin * price["prompt"] + tout * price["completion"])), 4)
    # END SOLUTION


# %%
try:
    assert cost_per_1000([{"prompt_tokens": 200, "completion_tokens": 4}], {"prompt": 1e-7, "completion": 4e-7}) == 0.0216
except NotImplementedError:
    def cost_per_1000(rows, price):
        return round(float(1000 * (np.mean([r["prompt_tokens"] or 0 for r in rows]) * price["prompt"]
                                   + np.mean([r["completion_tokens"] or 0 for r in rows]) * price["completion"])), 4)

base_latency = json.loads(G.supplied("baseline_latency.json", "D1_4").read_text(encoding="utf-8"))
ops = {"baseline": {**base_latency, "usd_per_1000": 0.0, "priced_as": "your own processor"}}
for name, model in MODELS.items():
    p = G.prices(model)
    paid = p["paid_endpoint"] or p["model"]
    ops[name] = {**G.percentiles([r["latency_s"] for r in runs[name]]), "usd_per_1000": cost_per_1000(runs[name], paid),
                 "priced_as": p["paid_id"] or model}
TABLE = QUALITY.join(pd.DataFrame(ops).T)
print(TABLE.to_string())
out = G.lab_dir("D2_4")
TABLE.to_csv(out / "comparison.csv")
G.keep("d24_table", TABLE.reset_index().to_dict(orient="records"))
if G.standin():
    print("\n⚠️  Stand-in run: the two 'model' rows are a keyword script. Their F1, latency and tokens say nothing about a model.")

# %% [markdown]
# ✏️ **Decide.** Sawaqi's budget is 300 ms at the 95th percentile per message, at 60,000 messages a month. Write
# three sentences in `artifacts/D2_4/decision.md`: which approach routes today, what it costs a month, and
# which variety you would send to a person until Day 3's model exists.

# %%
BUDGET_MS, MONTHLY = 300, 60_000
within = TABLE[TABLE["p95_ms"].astype(float) <= BUDGET_MS]
print("inside the latency budget:", list(within.index))
print("cost a month:", {n: round(float(r["usd_per_1000"]) * MONTHLY / 1000, 2) for n, r in TABLE.iterrows()})
(out / "decision.md").write_text("# Routing today: a decision\n\n1. ...\n2. ...\n3. ...\n", encoding="utf-8")

# %% [markdown]
# ## ✱ Stretch · A cascade: the baseline first, the model only when it is unsure
#
# Send a message to the larger model only when the baseline's confidence is below a threshold. For thresholds
# 0.4, 0.5 and 0.6: macro F1 of the cascade on the 40 messages, the share of messages that reach the model,
# and the cost per thousand.

# %%
# BEGIN SOLUTION
conf = baseline.loc[SAMPLE["msg_id"], "confidence"].to_numpy()
for th in (0.4, 0.5, 0.6):
    pred = [m if c < th else b for b, m, c in zip(PRED["baseline"], PRED["larger"], conf)]
    f1 = G.by_variety(SAMPLE["department"], pred, SAMPLE["variety"]).iloc[0]["macro_f1"]
    share = float((conf < th).mean())
    print(f"threshold {th}: macro F1 {f1}, {share:.0%} of messages reach the model, {share * ops['larger']['usd_per_1000']:.4f} USD per 1,000")
# END SOLUTION

# %% [markdown]
# ## Checkpoint
#
# 1. A cheaper model matches the larger one on English and loses on Arabizi. What decides?
# 2. Why price a free model at its paid endpoint?
# 3. What would change your decision: a number from this table, or a number that is not in it?
#
# <details><summary>Answers</summary>
#
# 1. Who writes to Sawaqi. If a tenth of the customers write Arabizi, the cheaper model fails a tenth of them
#    more often: decide on the variety scores weighted by the real mix, or route by variety.
# 2. A free endpoint has a daily request limit and no service promise. The cost of running the same model for
#    60,000 messages a month is the paid price.
# 3. Both. From the table: F1 by variety against the latency budget. Not in it: the licence of the weights,
#    where the provider processes the data, and what happens when the model is withdrawn.
#
# </details>
#
# **Carry forward.** `artifacts/D2_4/comparison.csv`: two of the three rows of the course's final comparison.
# Day 3 adds the fine-tuned encoder.
