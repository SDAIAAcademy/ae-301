"""Two labellers, one guideline: the data and every number of the agreement material in Day 3, Session 1.

    uv run python _tools/double_labels.py          # print every number
    uv run python _tools/double_labels.py --write  # also write data/double_labels.csv and its reference result

Three parts, each with its own fixed seed:

1. The lab (Lab 3.1, second stretch). 120 messages from the unlabelled pool, routed by two labellers
   under guideline v1, then again under guideline v2. Labeller B reads v1 differently from A in two
   ways the data plants: a meters message that mentions the bill becomes billing, and a message with
   a second clause from another department gets that department. Both also slip now and then.
   v2 adds two rules; most of the two patterns disappear.
2. The slides (parallel example, not the lab's answer): complaint types within billing and payments,
   60 messages, where "money left the card but the bill shows unpaid" splits the labellers.
3. The practice question: a two-type table for the leaks team, from fixed counts.

The labellers are simulated: the point is to read the agreement table, not to model people.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))

import pandas as pd  # noqa: E402

from prepare_data import ASIDES, DEPARTMENTS, TEMPLATES, VARIETIES  # noqa: E402

DATA = LABS / "data"
BILL_WORDS = ("فاتور", "fatoor", "fatur", "bill")


def normalise(text: str) -> str:
    sys.path.insert(0, str(LABS))
    from ghayth import normalise as n
    return n(text)


def kappa(a: list, b: list) -> dict:
    """Raw agreement, chance agreement and Cohen's kappa for two lists of labels."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n / n
    return {"n": n, "observed": round(po, 3), "chance": round(pe, 3), "kappa": round((po - pe) / (1 - pe), 3)}


def top_pairs(a: list, b: list, k: int = 3) -> list:
    pairs = Counter(tuple(sorted((x, y))) for x, y in zip(a, b) if x != y)
    return [[x, y, c] for (x, y), c in pairs.most_common(k)]


def aside_of(text: str, variety: str, gold: str) -> str | None:
    """The department whose words a second clause borrows, when the message carries one."""
    words = set(normalise(text).split())
    vi = 1 + VARIETIES.index(variety)
    for dept, *phrases in ASIDES:
        aw = set(normalise(phrases[vi - 1]).split())
        if dept != gold and len(aw & words) / len(aw) >= 0.6:
            return dept
    return None


def lab_part(write: bool) -> None:
    rng = random.Random(3101)
    m = pd.read_csv(DATA / "messages.csv").merge(pd.read_csv(DATA / "split.csv"), on="customer_id")
    pool = m[m["split"] == "pool"].sample(n=120, random_state=3101).reset_index(drop=True)

    def slip(label: str, p: float) -> str:
        return rng.choice([d for d in DEPARTMENTS if d != label]) if rng.random() < p else label

    rows = []
    for r in pool.itertuples():
        gold, text = r.department, r.text
        aside = aside_of(text, r.variety, gold)
        bill_meter = gold == "meters" and any(w in normalise(text) for w in BILL_WORDS) and aside != "billing"
        # guideline v1: "label the department the message is about"
        a1 = aside if (aside and rng.random() < 0.15) else gold
        b1 = "billing" if (bill_meter and rng.random() < 0.8) else (aside if (aside and rng.random() < 0.45) else gold)
        a1, b1 = slip(a1, 0.03), slip(b1, 0.03)
        # guideline v2 adds two rules: a bill blamed on the meter goes to meters; a clause about another,
        # past matter is not the request
        a2 = aside if (aside and rng.random() < 0.03) else gold
        b2 = "billing" if (bill_meter and rng.random() < 0.1) else (aside if (aside and rng.random() < 0.05) else gold)
        a2, b2 = slip(a2, 0.02), slip(b2, 0.02)
        rows.append({"msg_id": r.msg_id, "a_v1": a1, "b_v1": b1, "a_v2": a2, "b_v2": b2,
                     "_aside": aside or "", "_bill_meter": bill_meter, "_gold": gold})
    df = pd.DataFrame(rows)
    v1, v2 = kappa(list(df.a_v1), list(df.b_v1)), kappa(list(df.a_v2), list(df.b_v2))
    ref = {"v1": v1 | {"pairs": top_pairs(list(df.a_v1), list(df.b_v1))},
           "v2": v2 | {"pairs": top_pairs(list(df.a_v2), list(df.b_v2))}}
    print("LAB · 120 pool messages ·", int(df._aside.ne("").sum()), "with a second clause ·", int(df._bill_meter.sum()), "meters messages that mention the bill")
    print(json.dumps(ref, indent=1))
    disagree = df[df.a_v1 != df.b_v1]
    cause = disagree.apply(lambda r: "bill+meter" if r._bill_meter else ("second clause" if r._aside else "slip"), axis=1)
    print("v1 disagreements by planted cause:", cause.value_counts().to_dict())
    if write:
        df[["msg_id", "a_v1", "b_v1", "a_v2", "b_v2"]].to_csv(DATA / "double_labels.csv", index=False, lineterminator="\n")
        (DATA / "reference" / "d31_agreement.json").write_text(json.dumps(ref, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        print("wrote data/double_labels.csv and data/reference/d31_agreement.json")


def slide_part() -> None:
    """Complaint types in billing and payments: 60 messages, two labellers, before and after one rule."""
    rng = random.Random(3102)
    types = ["high_bill", "duplicate_charge", "payment_failed"]
    # gold mix of the 60 messages: what the contact centre sees in these two departments
    gold = ["high_bill"] * 30 + ["duplicate_charge"] * 12 + ["payment_failed"] * 18
    rng.shuffle(gold)
    # 8 of the 18 failed payments read "the amount left my card but the bill is still unpaid"
    card_taken = set(rng.sample([i for i, g in enumerate(gold) if g == "payment_failed"], 8))

    def label(i: int, g: str, p_card: float, p_slip: float) -> str:
        if i in card_taken and rng.random() < p_card:
            return "duplicate_charge"
        return rng.choice([t for t in types if t != g]) if rng.random() < p_slip else g

    for name, p_card_c, p_card_d in [("v1", 0.0, 0.85), ("v2", 0.0, 0.1)]:
        c = [label(i, g, p_card_c, 0.04) for i, g in enumerate(gold)]
        d = [label(i, g, p_card_d, 0.04) for i, g in enumerate(gold)]
        table = pd.crosstab(pd.Series(c, name="C"), pd.Series(d, name="D")).reindex(index=types, columns=types, fill_value=0)
        print(f"\nSLIDES · {name}")
        print(table.to_string())
        print(kappa(c, d), top_pairs(c, d))


def bank_part() -> None:
    """Leaks team, 100 messages, two types: fixed counts."""
    a = ["leak_street"] * 62 + ["leak_meter"] * 20 + ["leak_street"] * 4 + ["leak_meter"] * 14
    b = ["leak_street"] * 62 + ["leak_meter"] * 20 + ["leak_meter"] * 4 + ["leak_street"] * 14
    print("\nBANK · leaks team, 100 messages:", kappa(a, b), top_pairs(a, b))


if __name__ == "__main__":
    lab_part("--write" in sys.argv)
    slide_part()
    bank_part()
