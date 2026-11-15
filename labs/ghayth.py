"""Shared plumbing for the AE-301 labs: Ghayth, the language layer of Sawaqi Water Company.

The concepts are taught in the notebooks; this module holds the repeated parts: paths, quick
mode, loading the case data, reference results for the safety nets, scores by variety, and one
cached call to a hosted model (chat or embeddings) with a scripted stand-in for when no key works.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data import DEPARTMENTS, DISTRICTS, TYPES, VARIETIES

LABS = Path(__file__).resolve().parent
DATA = LABS / "data"
MODELS = LABS / "models"
REFERENCE = DATA / "reference"
ARTIFACTS = LABS / "artifacts"
CACHE = LABS / "cache"

try:
    from dotenv import load_dotenv
    load_dotenv(LABS / ".env")
except ImportError:
    pass

QUICK = os.getenv("ANLP_QUICK", "0") == "1"
PROVIDER_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_API_KEY", "")
CHAT_MODEL = os.getenv("ANLP_MODEL", "google/gemma-4-31b-it:free")
SMALL_MODEL = os.getenv("ANLP_SMALL_MODEL", "liquid/lfm-3-1.7b:free")
EMBED_MODEL = os.getenv("ANLP_EMBED_MODEL", "nvidia/nemotron-3.5-embed-1b:free")
FALLBACKS = [m for m in os.getenv("ANLP_FALLBACKS", "google/gemma-4-26b-a4b-it:free").split(",") if m]
USE_CACHE = os.getenv("ANLP_CACHE", "1") == "1"
_STANDIN = os.getenv("ANLP_STANDIN", "0") == "1" or not API_KEY
MIN_INTERVAL = float(os.getenv("ANLP_MIN_INTERVAL", "3.1"))  # seconds between uncached calls to a ":free" model


def pick(full, quick):
    """The small value in quick mode (ANLP_QUICK=1), the full one otherwise."""
    return quick if QUICK else full


def seed(n: int = 301) -> None:
    random.seed(n)
    np.random.seed(n)
    try:
        import torch
        torch.manual_seed(n)
    except ImportError:
        pass


def standin() -> bool:
    """True when calls go to the scripted stand-in instead of a hosted model."""
    return _STANDIN


def banner(lab: str) -> None:
    mode = "STAND-IN (scripted, not a model: its numbers say nothing about a model)" if _STANDIN else f"{PROVIDER_URL} · {CHAT_MODEL}"
    print(f"{lab} · quick mode: {QUICK} · model access: {mode}")


def lab_dir(name: str) -> Path:
    path = ARTIFACTS / name
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------- data ----------

def messages(split: str | None = None) -> pd.DataFrame:
    """The 2,000 messages with their split by customer; empty account and district become None.

    Splits: train (about 400 messages the contact centre labelled), val, test, and pool (the rest of
    the archive: its text is used for search and pretraining, its routing labels are not used).
    """
    m = pd.read_csv(DATA / "messages.csv", dtype={"account_number": str, "district": str})
    m = m.merge(pd.read_csv(DATA / "split.csv"), on="customer_id")
    m = m.astype(object).where(m.notna(), None)
    m["template"] = m["template"].astype(int)
    return m if split is None else m[m["split"] == split].reset_index(drop=True)


def labelled() -> pd.DataFrame:
    """The 1,100 messages with a routing label: train (400), val (300) and test (400), split by customer."""
    m = messages()
    return m[m["split"] != "pool"].reset_index(drop=True)


def jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def rules() -> dict[str, str]:
    """Rule id -> the rule's Markdown text (English and Arabic)."""
    return {p.name[:4]: p.read_text(encoding="utf-8") for p in sorted((DATA / "rules").glob("R-*.md"))}


def district_names() -> dict[str, list[str]]:
    """District key -> the ways it is written (Arabic, Arabizi with el and al, English)."""
    out = {}
    for key, (ar, lat, en) in DISTRICTS.items():
        out[key] = [ar, lat, lat.replace("el ", "al "), en, en.replace("Al-", "Al ")]
    return out


# ---------- Arabic normalisation (the reference version of Lab 1.1's exercise) ----------

_DIAC = re.compile("[ً-ْٰـ]")  # harakat, dagger alef, tatweel
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def normalise(text: str) -> str:
    text = _DIAC.sub("", text or "")
    text = re.sub("[أإآٱ]", "ا", text)
    text = text.replace("ة", "ه").replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    return text.translate(_DIGITS).lower()


# ---------- reference results for the safety nets ----------

def reference(name: str):
    return json.loads((REFERENCE / f"{name}.json").read_text(encoding="utf-8"))


def keep(name: str, value) -> None:
    """Writes a reference result when ANLP_WRITE_REF=1 (the course team's solution run)."""
    if os.getenv("ANLP_WRITE_REF") == "1":
        REFERENCE.mkdir(parents=True, exist_ok=True)
        (REFERENCE / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


def attempt(fn, name: str):
    """Runs an exercise. When it is not written yet, returns the course's reference result so later cells run."""
    try:
        value = fn()
    except NotImplementedError:
        print(f"↩️  Exercise not done yet: the cells below use the reference result '{name}'.")
        return reference(name)
    keep(name, value)
    return value


def check(ok: bool, what: str) -> None:
    print(("✅ " if ok else "❌ ") + what)


# ---------- tokenisers from the Hub (vocabulary files, not models) ----------

HUB_TOKENISERS = {
    "gpt2": ("openai-community/gpt2", "tokenizer.json"),  # byte-level BPE, trained on English web text
    "qwen3": ("Qwen/Qwen3-0.6B", "tokenizer.json"),  # byte-level BPE of a multilingual model family
    "camelbert": ("CAMeL-Lab/bert-base-arabic-camelbert-mix", "vocab.txt"),  # WordPiece, trained on Arabic
}


def hub_tokeniser(name: str):
    """One tokeniser file from the Hugging Face Hub (a few megabytes, cached). No model weights are fetched."""
    from huggingface_hub import hf_hub_download
    from tokenizers import BertWordPieceTokenizer, Tokenizer
    repo, file = HUB_TOKENISERS[name]
    path = hf_hub_download(repo, file)
    if file == "vocab.txt":
        return BertWordPieceTokenizer(path, lowercase=False, strip_accents=False)._tokenizer
    return Tokenizer.from_file(path)


def n_tokens(tokeniser, text: str) -> int:
    return len(tokeniser.encode(text, add_special_tokens=False).ids)


# ---------- scores ----------

def by_variety(y_true, y_pred, varieties) -> pd.DataFrame:
    """Macro F1 and accuracy overall and for each language variety."""
    from sklearn.metrics import accuracy_score, f1_score
    df = pd.DataFrame({"y": list(y_true), "p": list(y_pred), "v": list(varieties)})
    rows = [("all", len(df), f1_score(df.y, df.p, average="macro", zero_division=0), accuracy_score(df.y, df.p))]
    for v in VARIETIES:
        d = df[df.v == v]
        if len(d):
            rows.append((v, len(d), f1_score(d.y, d.p, average="macro", zero_division=0), accuracy_score(d.y, d.p)))
    return pd.DataFrame(rows, columns=["variety", "n", "macro_f1", "accuracy"]).round(3)


def percentiles(seconds) -> dict:
    a = np.asarray(seconds, dtype=float) * 1000
    return {"p50_ms": round(float(np.percentile(a, 50)), 2), "p95_ms": round(float(np.percentile(a, 95)), 2)}


# ---------- hosted models ----------

_client = None


def _openai():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(base_url=PROVIDER_URL, api_key=API_KEY, timeout=60, max_retries=2)
    return _client


def _key(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]


def _cached(kind: str, key: str):
    path = CACHE / kind / f"{key}.json"
    if USE_CACHE and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _store(kind: str, key: str, value) -> None:
    path = CACHE / kind / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


_last_call = 0.0


def _call(model: str, fn):
    """One request: paced for a free model, retried after a wait on a rate limit (429) or a busy server (5xx).

    Returns the response and the seconds the successful request took, without the waiting.
    """
    global _last_call
    for attempt in range(3):
        if model.endswith(":free"):  # free endpoints allow 20 requests a minute
            time.sleep(max(0.0, _last_call + MIN_INTERVAL - time.monotonic()))
        _last_call = time.monotonic()
        try:
            t0 = time.perf_counter()
            return fn(), time.perf_counter() - t0
        except Exception as err:  # noqa: BLE001
            status = getattr(err, "status_code", None) or 0
            if attempt == 2 or not (status == 429 or status >= 500):
                raise
            wait = 30 * (attempt + 1)
            print(f"  {model}: {'rate limit' if status == 429 else 'busy'} ({status}); waiting {wait} s, then trying again")
            time.sleep(wait)


def _fall_back(err: Exception) -> None:
    global _STANDIN
    print(f"⚠️  The hosted model refused ({type(err).__name__}: {str(err)[:120]}). Switching to the scripted stand-in.")
    _STANDIN = True


def chat(messages_: list[dict], model: str | None = None, temperature: float = 0.0, top_p: float = 1.0,
         max_tokens: int = 300, json_mode: bool = False, sample: int = 0, seed_: int | None = None) -> dict:
    """One chat call. Returns text, tokens in and out, latency in seconds, the model, and whether it was the stand-in.

    `sample` numbers repeated calls with the same settings, so the cache keeps each sample apart.
    """
    model = model or CHAT_MODEL
    req = {"model": model, "messages": messages_, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
           "json": json_mode, "sample": sample, "seed": seed_, "standin": _STANDIN}
    key = _key(req)
    hit = _cached("chat", key)
    if hit:
        return hit
    if not _STANDIN:
        for m in [model] + [f for f in FALLBACKS if f != model]:
            try:
                kw = {"response_format": {"type": "json_object"}} if json_mode else {}
                if seed_ is not None:
                    kw["seed"] = seed_
                r, seconds = _call(m, lambda: _openai().chat.completions.create(
                    model=m, messages=messages_, temperature=temperature, top_p=top_p, max_tokens=max_tokens, **kw))
                out = {"text": r.choices[0].message.content or "", "prompt_tokens": r.usage.prompt_tokens if r.usage else None,
                       "completion_tokens": r.usage.completion_tokens if r.usage else None,
                       "latency_s": round(seconds, 3), "model": m, "standin": False}
                _store("chat", key, out)
                return out
            except Exception as err:  # noqa: BLE001  a busy free model: try the next one
                status = getattr(err, "status_code", None)
                if status in (401, 403):
                    _fall_back(err)
                    break
                print(f"  {m} did not answer ({status or type(err).__name__}); trying the next model")
        else:
            _fall_back(RuntimeError("no model answered"))
    t0 = time.perf_counter()
    text = StandIn(model, temperature, sample).reply(messages_)
    prompt = " ".join(str(m["content"]) for m in messages_)
    return {"text": text, "prompt_tokens": len(prompt) // 4, "completion_tokens": len(text) // 4,
            "latency_s": round(time.perf_counter() - t0, 4), "model": model, "standin": True}


def embed(texts: list[str], model: str | None = None, batch: int = 64) -> np.ndarray:
    """Unit-length embeddings from a hosted embedding model, cached per text."""
    model = model or EMBED_MODEL
    started_real = not _STANDIN
    out, todo = [None] * len(texts), []
    for i, t in enumerate(texts):
        hit = _cached("embed", _key([model, t, _STANDIN]))
        if hit:
            out[i] = hit
        else:
            todo.append(i)
    for s in range(0, len(todo), batch):
        idx = todo[s:s + batch]
        vecs = None
        if not _STANDIN:
            try:
                r, _ = _call(model, lambda: _openai().embeddings.create(model=model, input=[texts[i] for i in idx]))
                vecs = [d.embedding for d in r.data]
            except Exception as err:  # noqa: BLE001
                _fall_back(err)
        if vecs is None:
            vecs = [_standin_vector(texts[i]) for i in idx]
        for i, v in zip(idx, vecs):
            out[i] = v
            _store("embed", _key([model, texts[i], _STANDIN]), v)
    if started_real and _STANDIN:  # the model stopped answering part-way: never mix its vectors with the stand-in's
        out = [_standin_vector(t) for t in texts]
    arr = np.asarray(out, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True).clip(1e-9)


def _standin_vector(text: str, dim: int = 384) -> list[float]:
    """Hashed character 3- to 5-grams: a lexical stand-in for an embedding model, nothing more."""
    v = np.zeros(dim, dtype=np.float32)
    t = f" {normalise(text)} "
    for n in (3, 4, 5):
        for i in range(len(t) - n + 1):
            h = int(hashlib.md5(t[i:i + n].encode()).hexdigest()[:8], 16)
            v[h % dim] += 1.0 if (h >> 20) & 1 else -1.0
    return v.tolist()


def prices(model: str) -> dict | None:
    """The provider's published price per token for a model, from OpenRouter's public model list.

    A free model has a price of 0; for the cost of the same model at volume, the paid endpoint
    of the same model (the id without ':free') is returned too, when it exists.
    """
    path = CACHE / "prices.json"
    try:
        if path.exists() and time.time() - path.stat().st_mtime < 86400:
            table = json.loads(path.read_text(encoding="utf-8"))
        else:
            import httpx
            data = httpx.get("https://openrouter.ai/api/v1/models", timeout=30).json()["data"]
            table = {m["id"]: {"prompt": float(m["pricing"]["prompt"]), "completion": float(m["pricing"]["completion"])} for m in data}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(table), encoding="utf-8")
    except Exception as err:  # noqa: BLE001
        print(f"⚠️  Could not read live prices ({type(err).__name__}); using the list saved in data/reference/prices.json.")
        table = reference("prices")["models"]
    paid = model.removesuffix(":free")
    return {"model": table.get(model), "paid_endpoint": table.get(paid), "paid_id": paid if paid in table else None}


# ---------- the scripted stand-in ----------

_KEYWORDS = {
    "billing": ["فاتور", "fatoor", "bill", "charged", "statement", "خصم", "khasam", "kashf", "مرتين", "martain", "twice"],
    "payments": ["سداد", "sadad", "دفع", "daf3", "pay", "تقسيط", "qasit", "instal", "arrears", "متأخر", "بطاق", "card"],
    "leaks": ["تسرب", "تسريب", "tasreeb", "leak", "pipe", "ماسور", "انبوب", "أنبوب", "كسر", "broken main", "drip"],
    "supply": ["انقطع", "مقطوع", "maqtoo3", "no water", "pressure", "ضغط", "6ag6", "خزان", "5azan", "tank"],
    "quality": ["لون", "loon", "brown", "yellow", "رائح", "ريح", "re7", "smell", "طعم", "taste", "رواسب", "sediment"],
    "meters": ["عداد", "3adad", "meter", "قراء", "qira2", "reading"],
    "connections": ["توصيل", "taw9eel", "connection", "نقل", "anqil", "transfer", "اغلاق", "إغلاق", "close my account", "tenant"],
    "wastewater": ["صرف", "مجاري", "majari", "sewage", "sewer", "انسداد", "insidad", "drain", "بيار", "septic", "manhole", "منهل"],
}
_TYPE_HINTS = {
    "high_bill": ["غالي", "مرتفع", "ghalya", "higher", "high", "dispute", "اعترض", "معترض"], "duplicate_charge": ["مرتين", "martain", "twice", "مكرر"],
    "payment_failed": ["فشل", "خطأ", "failed", "unpaid", "options"], "installment_request": ["تقسيط", "اقسط", "qasit", "instal", "arrears", "plan", "جدول"],
    "leak_street": ["شارع", "share3", "street", "pipe", "park", "حديق", "تقاطع"], "leak_meter": ["عداد", "3adad", "meter"],
    "no_water": ["مقطوع", "انقطع", "maqtoo3", "no water", "come back", "فاضي", "empty"], "low_pressure": ["ضغط", "6ag6", "pressure", "ضعيف"],
    "discoloured": ["لون", "loon", "brown", "yellow", "رواسب", "sediment"], "bad_smell": ["رائح", "ريح", "re7", "smell", "طعم", "taste"],
    "meter_fault": ["واقف", "شاشة", "فحص", "test", "screen", "move", "خربان", "معطل", "broken"], "meter_reading": ["قراء", "qira2", "reading", "reader"],
    "new_connection": ["توصيل", "taw9eel", "connection", "new meter", "عداد جديد"], "ownership_transfer": ["نقل", "انقل", "anqil", "transfer", "close", "اغلاق", "إغلاق", "tenant"],
    "sewage_overflow": ["طفح", "طافح", "6af7a", "overflow", "فايض", "تفيض", "راكد", "واقفه"], "sewage_blockage": ["انسداد", "insidad", "blocked", "drain", "شفط", "suction", "تنزل", "تنصرف"],
}


def keyword_type(text: str) -> str:
    """A complaint type from keyword lists: the department first, then the better-matching of its two types."""
    dept, _ = keyword_department(text)
    t = normalise(text)
    return max([c for c, d in TYPES.items() if d == dept], key=lambda c: sum(normalise(k) in t for k in _TYPE_HINTS[c]))


def keyword_department(text: str) -> tuple[str, int]:
    t = normalise(text)
    scores = {d: sum(normalise(k) in t for k in ks) for d, ks in _KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best, scores[best]


class StandIn:
    """A scripted responder that imitates a chat model for the course's prompts. It is not a model."""

    def __init__(self, model: str, temperature: float, sample: int):
        self.model, self.t, self.sample = model, temperature, sample

    def rng(self, text: str) -> random.Random:
        return random.Random(_key([self.model, text, self.t, self.sample]))

    def reply(self, messages_: list[dict]) -> str:
        system = " ".join(m["content"] for m in messages_ if m["role"] == "system")
        user = [m["content"] for m in messages_ if m["role"] == "user"][-1]
        if "account_number" in system and "JSON" in system:
            if user.startswith("Your reply failed validation"):
                return self.repair(messages_)
            return self.extract(user)
        if "Departments:" in system:
            return self.route(user)
        if "RULES" in user or "RULES" in system:
            return self.ground(user)
        return self.write_reply(user)

    def route(self, text: str) -> str:
        msg = text.split("Message:")[-1]
        dept, score = keyword_department(msg)
        rng = self.rng(msg)
        if score == 0 or rng.random() < (0.12 if "1.7b" in self.model else 0.05):
            dept = rng.choice(DEPARTMENTS)
        return dept if rng.random() < 0.8 else f"Department: {dept}"

    def extract(self, text: str) -> str:
        msg = text.split("Message:")[-1]
        rng = self.rng(msg)
        acc = re.search(r"\[ACCOUNT_\d+\]|\b30\d{8}\b", msg)
        district = None
        for key, names in district_names().items():
            if any(normalise(n) in normalise(msg) for n in names):
                district = key
        dept, _ = keyword_department(msg)
        ctype = keyword_type(msg)
        account = acc.group(0) if acc else (("30" + "".join(rng.choice("0123456789") for _ in range(8))) if rng.random() < 0.3 else None)
        out = json.dumps({"account_number": account, "district": district, "complaint_type": ctype}, ensure_ascii=False)
        roll = rng.random()
        if roll < 0.08:
            return "Here is the JSON you asked for:\n```json\n" + out + "\n```"
        if roll < 0.12:
            return out.replace('"complaint_type"', '"type"')
        return out

    def repair(self, messages_: list[dict]) -> str:
        """A second try after a validation error: it fixes the format and drops a value it invented, most of the time."""
        first = [m["content"] for m in messages_ if m["role"] == "user"][0]
        bad = [m["content"] for m in messages_ if m["role"] == "assistant"][-1]
        if self.rng(first + "repair").random() < 0.2:
            return bad
        m = re.search(r"\{.*\}", bad, flags=re.S)
        obj = json.loads(m.group(0)) if m else {}
        if "type" in obj:
            obj["complaint_type"] = obj.pop("type")
        if obj.get("account_number") and str(obj["account_number"]) not in first:
            obj["account_number"] = None
        return json.dumps(obj, ensure_ascii=False)

    def ground(self, text: str) -> str:
        text, _, question = text.partition("QUESTION:")
        question = question.strip()
        chunks = re.findall(r"\[(R-\d\d)\]\s*(.+?)(?=\n\[R-\d\d\]|\Z)", text, flags=re.S)
        qwords = {w for w in re.findall(r"\w{3,}", normalise(question))}
        best, best_overlap = None, 0
        for rid, body in chunks:
            overlap = len(qwords & set(re.findall(r"\w{3,}", normalise(body))))
            if overlap > best_overlap:
                best, best_overlap = (rid, body), overlap
        if not best or best_overlap < 2:
            return "NOT_IN_RULES"
        sentences = re.split(r"(?<=[.!؟?])\s+", best[1].strip())
        scored = sorted(sentences, key=lambda s: -len(qwords & set(re.findall(r"\w{3,}", normalise(s)))))
        return f"{scored[0].strip()} [{best[0]}]"

    def write_reply(self, text: str) -> str:
        rng = self.rng(text)
        facts = re.findall(r"SW-\d+", text)
        ticket = facts[0] if facts else "your request"
        openers = ["Thank you for contacting Sawaqi.", "We are sorry for the trouble.", "Thanks for letting us know."]
        middles = [f"Your request is logged as {ticket}.", f"We have opened {ticket} for you.", f"Reference: {ticket}."]
        k = 0 if self.t == 0 else None
        reply = " ".join([openers[k if k is not None else rng.randrange(3)], middles[k if k is not None else rng.randrange(3)]])
        hours = re.search(r'"crew_eta_hours": (\d+)', text)
        if hours:
            reply += f" A crew will reach you within {hours.group(1)} hours."
        if rng.random() < 0.15 + 0.45 * self.t:  # invented details grow with temperature
            reply += rng.choice([" You will receive a credit of 50 SAR.", " The work will finish within 48 hours.",
                                 " Your engineer, Ahmed, will call you on 0500000000.", " This is covered under rule R-11."])
        return reply


# ---------- files a later lab starts from ----------

def keep_file(path: Path) -> None:
    """Copies a lab's output into data/reference/ when ANLP_WRITE_REF=1 (the course team's solution run)."""
    if os.getenv("ANLP_WRITE_REF") == "1":
        REFERENCE.mkdir(parents=True, exist_ok=True)
        (REFERENCE / path.name).write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def supplied(name: str, lab: str | None = None) -> Path:
    """Your own output of an earlier lab when it exists, otherwise the supplied copy in data/reference/."""
    own = ARTIFACTS / lab / name if lab else None
    if own is not None and own.exists():
        print(f"using your own {name} from Lab {lab}")
        return own
    return REFERENCE / name


def hub_tokenisers() -> dict:
    """The three tokenisers of Lab 1.1; one that cannot be fetched is left out with a warning."""
    out = {}
    for name in HUB_TOKENISERS:
        try:
            out[name] = hub_tokeniser(name)
        except Exception as err:  # noqa: BLE001
            print(f"⚠️  Could not fetch the {name} tokeniser ({type(err).__name__}); the reference counts cover it.")
    return out


def baseline():
    """Day 1's routing baseline, not yet fitted: TF-IDF of words and character n-grams, then a linear model."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline, make_union
    return make_pipeline(
        make_union(TfidfVectorizer(preprocessor=normalise, sublinear_tf=True, ngram_range=(1, 2)),
                   TfidfVectorizer(preprocessor=normalise, sublinear_tf=True, analyzer="char_wb", ngram_range=(2, 4))),
        LogisticRegression(C=10, max_iter=1000))
