"""Ghayth as a service: the starter for the capstone (Lab 3.4 as one file).

    cd labs
    uv run uvicorn capstone.service:app --port 8000

Then open the demo page at the address uvicorn prints, or the generated API documentation under /docs.
It serves the Day 1 baseline; set GHAYTH_MODEL=encoder to serve the supplied fine-tuned encoder. Replace
`load_scorer` by your own model, keep the hand-off rule honest, and keep the smoke test passing:

    uv run python capstone/evaluate.py --smoke
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ghayth as G  # noqa: E402
import tinytf as T  # noqa: E402

CHOSEN = os.getenv("GHAYTH_MODEL", "baseline")
THRESHOLDS = {"arabic": 0.5, "en": 0.5, "arabizi": 0.7}
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def load_scorer(kind: str):
    """Returns a function from a text to {department: probability}."""
    if kind == "baseline":
        train = G.messages("train")
        model = G.baseline().fit(train["text"], train["department"])
        return lambda text: dict(zip(model.classes_, model.predict_proba([text])[0].tolist()))
    model, labels = T.load_classifier(G.MODELS / "ghayth-enc-routing-full.pt")
    tok = T.load_tokenizer()

    def score(text: str) -> dict:
        with torch.no_grad():
            return dict(zip(labels, model(T.encode(tok, [text])).softmax(-1)[0].tolist()))
    return score


SCORE = load_scorer(CHOSEN)


def variety_of(text: str) -> str:
    """A rough guess from the script: 'arabic' (MSA or Gulf), 'arabizi' or 'en'."""
    letters = [ch for ch in text if ch.isalpha()]
    if letters and sum("؀" <= ch <= "ۿ" for ch in letters) / len(letters) > 0.5:
        return "arabic"
    hints = re.findall(r"\b\w*[a-z][235679]\w*|\b[235679][a-z]\w*|\b(?:el|al|fi|min|mai|may|abi|wesh)\b", text.lower())
    return "arabizi" if len(hints) >= 2 else "en"


def fields(text: str) -> dict:
    """Account number and district by exact patterns. Replace or extend with your validated extraction."""
    plain = text.translate(DIGITS)
    account = re.search(r"(?<!\d)30\d{8}(?!\d)", plain)
    district = next((key for key, names in G.district_names().items() if any(G.normalise(n) in G.normalise(plain) for n in names)), None)
    return {"account_number": account.group(0) if account else None, "district": district}


def decide(text: str) -> dict:
    variety = variety_of(text)
    if len(re.findall(r"[^\W\d_]+", text)) < 2:
        return {"department": None, "confidence": 0.0, "variety": variety, "handoff": True, "reason": "no request",
                "account_number": None, "district": None}
    ranked = sorted(SCORE(text).items(), key=lambda kv: -kv[1])
    (best, p1), (_, p2) = ranked[0], ranked[1]
    reason = "low confidence" if p1 < THRESHOLDS[variety] else "two departments" if p2 >= 0.6 * p1 else None
    return {"department": best, "confidence": round(p1, 3), "variety": variety, "handoff": reason is not None, "reason": reason,
            **fields(text)}


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
<p>Synthetic messages only. Do not paste a real customer's message.</p>
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
