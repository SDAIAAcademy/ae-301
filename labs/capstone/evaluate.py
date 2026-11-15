"""Score a routing model by language variety, and write a timestamped log of the run.

    cd labs
    uv run python capstone/evaluate.py --smoke                          # the 24-message smoke test against capstone/service.py
    uv run python capstone/evaluate.py --batch capstone/batch.example.csv   # re-score the service on a batch (the defence)
    uv run python capstone/evaluate.py --predictions my_predictions.csv     # score a file you made yourself

A batch is a CSV with the columns msg_id, text, variety, department. A predictions file has msg_id, variety,
y_true, y_pred. Every run writes capstone/logs/evaluation_<UTC time>.json: commit those logs, they are the
evidence that the numbers in your README were measured.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ghayth as G  # noqa: E402


def client():
    from fastapi.testclient import TestClient

    from capstone.service import app
    return TestClient(app)


def smoke() -> dict:
    c, failures = client(), []
    for case in G.jsonl("smoke_test.jsonl"):
        r = c.post("/route", json={"text": case["text"]})
        if r.status_code != 200:
            failures.append({"id": case["id"], "why": "status"})
        elif case["handoff"] and not r.json()["handoff"]:
            failures.append({"id": case["id"], "why": "edge case not handed off"})
        elif not case["handoff"] and r.json()["department"] not in G.DEPARTMENTS:
            failures.append({"id": case["id"], "why": "no department"})
    print("smoke test:", "PASS" if not failures else f"FAIL {failures}")
    return {"kind": "smoke", "cases": 24, "failures": failures}


def score(df: pd.DataFrame, kind: str) -> dict:
    table = G.by_variety(df["y_true"], df["y_pred"].fillna("none"), df["variety"])
    print(table.to_string(index=False))
    wrong = df[df["y_true"] != df["y_pred"]]
    print(f"\n{len(wrong)} errors. The first five (explain two of them in your defence):")
    for row in wrong.head(5).itertuples():
        print(f"  {row.msg_id} [{row.variety}] true {row.y_true} → predicted {row.y_pred}" + (f" | {row.text[:90]}" if hasattr(row, "text") else ""))
    return {"kind": kind, "n": len(df), "by_variety": table.to_dict(orient="records"), "errors": list(wrong["msg_id"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch")
    ap.add_argument("--predictions")
    args = ap.parse_args()
    if args.smoke:
        result = smoke()
    elif args.batch:
        batch = pd.read_csv(args.batch, dtype=str).fillna("")
        c = client()
        answers = [c.post("/route", json={"text": t}).json() for t in batch["text"]]
        df = batch.rename(columns={"department": "y_true"}).assign(y_pred=[a["department"] for a in answers],
                                                                  handoff=[a["handoff"] for a in answers])
        print(f"handed off: {int(df['handoff'].sum())} of {len(df)}")
        result = score(df, f"batch:{Path(args.batch).name}")
    elif args.predictions:
        result = score(pd.read_csv(args.predictions, dtype=str), f"predictions:{Path(args.predictions).name}")
    else:
        ap.error("choose --smoke, --batch or --predictions")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (HERE / "logs").mkdir(exist_ok=True)
    out = HERE / "logs" / f"evaluation_{stamp}.json"
    out.write_text(json.dumps({"utc": stamp, **result}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nlog:", out.relative_to(HERE.parent))


if __name__ == "__main__":
    main()
