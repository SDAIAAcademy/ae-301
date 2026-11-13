# %% [markdown]
# # Lab 1.4 · A routing baseline, without AI
#
# Day 1 · Session 4 (Decision models) · 25 min. **AI assistants off for this lab.** Hessa needs every message
# sent to one of eight departments. Before any large model, you build the cheapest thing that could work: word
# and character counts and a linear model. Its score by variety is the number every later model has to beat,
# and you write the metric yourself, because the supervised check will ask you to.
#
# **Reference state.** `data/messages.csv` with the split by customer in `data/split.csv`: about 400 messages
# the contact centre labelled (train), 300 for validation, 400 held out for the test. Nothing from another lab.
#
# **Tasks**
# - [ ] 1 · Features: word and character n-grams (6')
# - [ ] 2 · Train a linear model on the customer split (5')
# - [ ] 3 · Macro F1 overall and by dialect (7')
# - [ ] 4 · Save predictions and CPU latency (7')
# - [ ] ✱ Stretch · Abstain below a confidence threshold and count the hand-offs

# %%
#@setup
G.banner("Lab 1.4")

# %%
import time

import numpy as np
import pandas as pd

train, val, test = G.messages("train"), G.messages("val"), G.messages("test")
print("train", len(train), "| val", len(val), "| test", len(test))
print(pd.crosstab(train["department"], train["variety"])[G.VARIETIES].to_string())

# %% [markdown]
# 🔍 Read the Arabizi column before you train anything. How many labelled Arabizi messages does each department
# have? What do you expect for those departments?

# %% [markdown]
# ## 1 · Features: word and character n-grams (6')
#
# A bag of words counts words and forgets their order. Character n-grams inside word boundaries catch what
# words miss: a word with a typo, a prefix glued to it (الفاتوره, بالفاتوره), an Arabizi spelling that varies.
#
# ✏️ **Exercise** · Write `build_features()`: a `FeatureUnion` (use `make_union`) of two `TfidfVectorizer`s, both
# with `preprocessor=G.normalise` and `sublinear_tf=True`: words (`ngram_range=(1, 2)`) and characters
# (`analyzer="char_wb"`, `ngram_range=(2, 4)`).

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union


def build_features():
    # BEGIN SOLUTION
    return make_union(TfidfVectorizer(preprocessor=G.normalise, sublinear_tf=True, ngram_range=(1, 2)),
                      TfidfVectorizer(preprocessor=G.normalise, sublinear_tf=True, analyzer="char_wb", ngram_range=(2, 4)))
    # END SOLUTION


# %%
try:
    features = build_features()
except NotImplementedError:
    features = make_union(TfidfVectorizer(preprocessor=G.normalise, sublinear_tf=True, ngram_range=(1, 2)),
                          TfidfVectorizer(preprocessor=G.normalise, sublinear_tf=True, analyzer="char_wb", ngram_range=(2, 4)))
X = features.fit_transform(train["text"])
print("feature matrix:", X.shape, f"| {X.nnz / X.shape[0]:.0f} non-zero features per message")
assert X.shape[0] == len(train) and X.shape[1] > 5000

# %% [markdown]
# ## 2 · Train a linear model on the customer split (5')
#
# ✏️ **Exercise** · Write `train_baseline(df)`: a pipeline of `build_features()` and
# `LogisticRegression(C=10, max_iter=1000)`, fitted on the texts and departments of `df`.

# %%
def train_baseline(df: pd.DataFrame):
    # BEGIN SOLUTION
    return make_pipeline(build_features(), LogisticRegression(C=10, max_iter=1000)).fit(df["text"], df["department"])
    # END SOLUTION


# %%
try:
    model = train_baseline(train)
except NotImplementedError:
    model = make_pipeline(features, LogisticRegression(C=10, max_iter=1000)).fit(train["text"], train["department"])
pred = model.predict(test["text"])
print("test accuracy:", round(float((pred == test["department"]).mean()), 3))

# %% [markdown]
# ## 3 · Macro F1 overall and by dialect (7')
#
# Accuracy counts messages, so a large department hides a small one. Macro F1 gives every department the same
# weight: F1 per department, then the plain mean.
#
# ✏️ **Exercise** · Write `macro_f1(y_true, y_pred, labels)` by hand, without scikit-learn's metrics. For each
# label: precision = TP / (TP + FP), recall = TP / (TP + FN), F1 = 2PR / (P + R), and 0 when a denominator
# is 0. A label that appears in neither list is left out of the mean.

# %%
def macro_f1(y_true, y_pred, labels) -> float:
    # BEGIN SOLUTION
    y_true, y_pred, f1s = list(y_true), list(y_pred), []
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(f1s) / len(f1s)
    # END SOLUTION


# %%
from sklearn.metrics import f1_score

try:
    mine = macro_f1(test["department"], pred, G.DEPARTMENTS)
    assert abs(mine - f1_score(test["department"], pred, average="macro")) < 1e-9, "differs from scikit-learn"
    assert abs(macro_f1(["a", "a", "b"], ["a", "b", "b"], ["a", "b", "c"]) - 2 / 3) < 1e-9, "a label in neither list is left out"
    print("✅ your macro F1 matches scikit-learn:", round(mine, 3))
except NotImplementedError:
    print("↩️  macro_f1 not written yet: the table below uses scikit-learn")

scores = G.by_variety(test["department"], pred, test["variety"])
G.keep("d14_scores", scores.to_dict(orient="records"))
print(scores.to_string(index=False))
arabizi = test["variety"] == "arabizi"
print("\nArabizi test messages, by department: right / all")
print(pd.DataFrame({"all": test[arabizi].groupby("department").size(),
                    "right": test[arabizi & (pred == test["department"])].groupby("department").size()}).fillna(0).astype(int).T.to_string())
assert scores.iloc[0]["macro_f1"] > 0.7

# %% [markdown]
# 🔍 The first row would go on a slide. The Arabizi row is the one a customer feels. Which departments does an
# Arabizi message rarely reach, and how many labelled examples did they have in the table at the top?

# %% [markdown]
# ## 4 · Save predictions and CPU latency (7')
#
# Day 2 and Day 3 compare other models with this one, on the same test messages. Save the predictions with the
# model's confidence, and measure how long one message takes on this processor.
#
# ✏️ **Exercise** · Write `latency(model, texts, n)`: predict the first `n` texts one at a time, time each call
# with `time.perf_counter()`, and return the list of seconds.

# %%
def latency(model, texts, n: int = 200) -> list[float]:
    # BEGIN SOLUTION
    out = []
    for t in list(texts)[:n]:
        t0 = time.perf_counter()
        model.predict([t])
        out.append(time.perf_counter() - t0)
    return out
    # END SOLUTION


# %%
try:
    lat = G.percentiles(latency(model, test["text"], G.pick(200, 50)))
except NotImplementedError:
    lat = G.reference("baseline_latency")
    print("↩️  latency not written yet: using the reference measurement")
print("CPU latency per message:", lat)

out = G.lab_dir("D1_4")
proba = model.predict_proba(test["text"])
predictions = pd.DataFrame({"msg_id": test["msg_id"], "variety": test["variety"], "y_true": test["department"],
                            "y_pred": pred, "confidence": proba.max(1).round(4)})
predictions.to_csv(out / "baseline_predictions.csv", index=False)
(out / "baseline_latency.json").write_text(pd.Series(lat).to_json(), encoding="utf-8")
G.keep_file(out / "baseline_predictions.csv")
G.keep("baseline_latency", lat)
print("saved", len(predictions), "predictions to artifacts/D1_4/")

# %% [markdown]
# ## ✱ Stretch · Abstain below a confidence threshold and count the hand-offs
#
# A router may say "I do not know" and send the message to a person. For thresholds 0.3 to 0.7: how many test
# messages are handed off, and what is the accuracy on the ones the model keeps? Do the same for Arabizi only.

# %%
# BEGIN SOLUTION
rows = []
for th in (0.3, 0.4, 0.5, 0.6, 0.7):
    keep = predictions["confidence"] >= th
    kept = predictions[keep]
    az = kept[kept["variety"] == "arabizi"]
    rows.append({"threshold": th, "handed_off": int((~keep).sum()), "accuracy_on_kept": round(float((kept.y_true == kept.y_pred).mean()), 3),
                 "arabizi_handed_off": int(((~keep) & (predictions["variety"] == "arabizi")).sum()),
                 "arabizi_accuracy_on_kept": round(float((az.y_true == az.y_pred).mean()), 3)})
print(pd.DataFrame(rows).to_string(index=False))
# END SOLUTION

# %% [markdown]
# ## Checkpoint
#
# 1. Why is macro F1 for Arabizi so far below its accuracy?
# 2. The split is by customer. What would a random split have done to these numbers?
# 3. What does this baseline cost per thousand messages, and what would have to be true for a larger model to
#    be worth more?
#
# <details><summary>Answers</summary>
#
# 1. Accuracy is carried by the departments with many Arabizi messages. Macro F1 counts every department once,
#    and some departments have almost no Arabizi training messages, so their F1 is near zero.
# 2. Raised them: the same customer writes the same complaint again, and a random split puts one copy in
#    training and one in the test. Lab 3.1 measures by how much.
# 3. A fraction of a second of one processor core per thousand: next to nothing. A larger model must be clearly
#    better where this one fails (Arabizi, rare departments), within the latency budget.
#
# </details>
#
# **Carry forward.** `artifacts/D1_4/baseline_predictions.csv` and `baseline_latency.json`. Labs 2.4 and 3.4
# read them; when they are missing, they read the supplied copies in `data/reference/`.
