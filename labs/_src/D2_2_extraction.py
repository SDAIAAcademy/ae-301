# %% [markdown]
# # Lab 2.2 · Masked, validated extraction
#
# Day 2 · Session 2 (Prompting and structured outputs) · 25 min. Every routed message needs three fields: the
# account number, the district and the complaint type. A hosted model can read them from free text in four
# varieties, but Turki, the data protection officer, will not let a phone number or an account number leave
# Sawaqi. You mask first, extract into a schema, validate, repair once, log what still fails, and score each
# field against a regular-expression baseline.
#
# **Reference state.** `data/extraction_200.jsonl`: 200 test messages with gold fields (40 in quick mode,
# `ANLP_QUICK=1`, to stay inside a free daily quota). Nothing from another lab. The model is called through
# `G.chat` (cached); without a working key a scripted stand-in answers, and the banner says so.
#
# **Tasks**
# - [ ] 1 · Mask personal fields, with a test (6')
# - [ ] 2 · A schema whose validators catch wrong values (5')
# - [ ] 3 · Extract with a hosted model, repair once, log failures (8')
# - [ ] 4 · Field precision and recall against a regex baseline (6')
# - [ ] ✱ Stretch · Add two examples per variety to the prompt and score again

# %%
#@setup
G.banner("Lab 2.2")

# %%
import json
import re
from typing import Literal, Optional, get_args

import pandas as pd
from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator

import prepare_data as P

ITEMS = G.jsonl("extraction_200.jsonl")[: G.pick(200, 40)]
print(len(ITEMS), "messages |", pd.Series([i["variety"] for i in ITEMS]).value_counts().to_dict())
print(ITEMS[1]["text"], "\n ", ITEMS[1]["gold"])

# %% [markdown]
# ## 1 · Mask personal fields, with a test (6')
#
# Sawaqi's rule R-10 names customer names, phone numbers and account numbers as personal data, classifies
# customer records as Restricted and requires masking before message text goes to an external service. The
# model does not need the digits to find the account number: a placeholder in the same place is enough, and
# you put the real value back afterwards.
#
# Phone numbers are `05` and eight digits. Account numbers are `30` and eight digits. Both may be written in
# Arabic-Indic digits. Names come from the lists in `prepare_data.py` (in production: a name recogniser).
#
# ✏️ **Exercise** · Write `mask(text)`. Turn Arabic-Indic digits into Western digits, then replace every phone
# number with `[PHONE_1]`, `[PHONE_2]`..., every account number with `[ACCOUNT_1]`... and every known first or
# last name with `[NAME_1]`... Return the masked text and a dictionary from placeholder to original value.

# %%
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
NAMES = sorted(set(P.FIRST_AR + P.LAST_AR + P.FIRST_LAT + P.LAST_LAT), key=len, reverse=True)
NAME_RE = re.compile("|".join(re.escape(n) for n in NAMES))


def mask(text: str) -> tuple[str, dict]:
    # BEGIN SOLUTION
    mapping = {}

    def replacer(kind):
        def put(m):
            key = f"[{kind}_{sum(k.startswith('[' + kind) for k in mapping) + 1}]"
            mapping[key] = m.group(0)
            return key
        return put

    text = text.translate(DIGITS)
    text = re.sub(r"(?<!\d)05\d{8}(?!\d)", replacer("PHONE"), text)
    text = re.sub(r"(?<!\d)30\d{8}(?!\d)", replacer("ACCOUNT"), text)
    text = NAME_RE.sub(replacer("NAME"), text)
    return text, mapping
    # END SOLUTION


def unmask(value, mapping: dict):
    return mapping.get(value, value) if isinstance(value, str) else value


# %%
def reference_mask(text):
    mapping = {}
    text = text.translate(DIGITS)
    for kind, pattern in [("PHONE", r"(?<!\d)05\d{8}(?!\d)"), ("ACCOUNT", r"(?<!\d)30\d{8}(?!\d)"), ("NAME", NAME_RE.pattern)]:
        def put(m, kind=kind):
            key = f"[{kind}_{sum(k.startswith('[' + kind) for k in mapping) + 1}]"
            mapping[key] = m.group(0)
            return key
        text = re.sub(pattern, put, text)
    return text, mapping


try:
    mask("x")
except NotImplementedError:
    mask = reference_mask


def masking_test(items) -> dict:
    """Nothing personal may be left: no run of 10 digits, no known name. And unmasking restores the text."""
    leaks = restored = 0
    for it in items:
        masked, mapping = mask(it["text"])
        leaks += bool(re.search(r"\d{10}", masked) or NAME_RE.search(masked))
        back = masked
        for key, value in mapping.items():
            back = back.replace(key, value)
        restored += back == it["text"].translate(DIGITS)
    return {"messages": len(items), "with_a_leak": leaks, "restored": restored}


result = masking_test(ITEMS)
print(result)
assert result["with_a_leak"] == 0 and result["restored"] == len(ITEMS), "the masking test fails"
print("✅ masking test passes")
masked, mapping = mask(ITEMS[1]["text"])
print(masked, "\n ", mapping)

# %% [markdown]
# 🔍 The test is the evidence Turki asks for. It proves the patterns you wrote, on these 200 messages. What
# would get through that it cannot see? (A name that is not in the list, a phone number written with spaces.)

# %% [markdown]
# ## 2 · A schema whose validators catch wrong values (5')
#
# The schema is the contract between the model and your code. Types catch a wrong shape; **validators** catch a
# wrong value. The model saw masked text, so the only account number it can legitimately return is a
# placeholder that is in this message's mapping. Anything else was invented or copied from the wrong place.
#
# ✏️ **Exercise** · Complete the validator `known_account`: when the value is not `None`, it must be a key of
# `info.context["mapping"]` that starts with `[ACCOUNT_`; otherwise raise `ValueError` with a message that
# tells the model what to do.

# %%
District = Literal["ghadeer", "manhal", "shurfa", "wurood", "yasmeen", "qalaa", "rabwa", "naseem"]
ComplaintType = Literal["high_bill", "duplicate_charge", "payment_failed", "installment_request", "leak_street", "leak_meter",
                        "no_water", "low_pressure", "discoloured", "bad_smell", "meter_fault", "meter_reading",
                        "new_connection", "ownership_transfer", "sewage_overflow", "sewage_blockage"]


class Extraction(BaseModel, extra="forbid"):
    account_number: Optional[str]
    district: Optional[District]
    complaint_type: ComplaintType

    @field_validator("account_number")
    @classmethod
    def known_account(cls, value, info: ValidationInfo):
        # BEGIN SOLUTION
        mapping = (info.context or {}).get("mapping", {})
        if value is not None and not (value.startswith("[ACCOUNT_") and value in mapping):
            raise ValueError("account_number must be an [ACCOUNT_n] placeholder that appears in the message, or null")
        return value
        # END SOLUTION


# %%
def validate(text: str, mapping: dict) -> Extraction:
    try:
        return Extraction.model_validate_json(text, context={"mapping": mapping})
    except NotImplementedError:  # safety net: the same rule, written outside the class
        data = json.loads(text)
        account = data.get("account_number")
        if account is not None and not (account.startswith("[ACCOUNT_") and account in mapping):
            raise ValueError("account_number must be an [ACCOUNT_n] placeholder that appears in the message, or null")
        if set(data) != set(Extraction.model_fields):
            raise ValueError("the keys must be exactly: " + ", ".join(Extraction.model_fields))
        if data["district"] not in (None, *get_args(District)) or data["complaint_type"] not in get_args(ComplaintType):
            raise ValueError("district or complaint_type is not a value of the list")
        return Extraction.model_construct(**data)


good = '{"account_number": "[ACCOUNT_1]", "district": "naseem", "complaint_type": "meter_fault"}'
assert validate(good, mapping).district == "naseem"
for bad, why in [(good.replace("[ACCOUNT_1]", "3011111111"), "an account number the model never saw"),
                 (good.replace("[ACCOUNT_1]", "[PHONE_1]"), "a phone placeholder in the account field"),
                 (good.replace("naseem", "Al-Naseem"), "a district outside the list"),
                 (good.replace("complaint_type", "type"), "a wrong key")]:
    try:
        validate(bad, mapping)
        raise AssertionError(f"accepted {why}")
    except (ValidationError, ValueError) as err:
        if isinstance(err, AssertionError):
            raise
        print("✅ rejected:", why)

# %% [markdown]
# ## 3 · Extract with a hosted model, repair once, log failures (8')
#
# The loop: mask → ask (temperature 0, JSON mode) → validate → on a failure, send the error back **once** →
# validate again → if it still fails, log it and return nothing. A logged failure goes to a person; it is
# never guessed.
#
# ✏️ **Exercise** · Complete `extract(item)`. Use `G.chat(messages, json_mode=True, max_tokens=150)`. For the
# repair call, append the model's reply as an `assistant` message and a `user` message that starts with
# `"Your reply failed validation: "` followed by the error, and ask for the corrected JSON only. Append one
# entry to `FAILURES` for every failed validation: `msg_id`, `attempt` (1 or 2) and `error`.

# %%
SYSTEM = (
    "You extract fields from one customer message of a water utility. The message may be in Modern Standard Arabic, "
    "Gulf dialect, Arabic in Latin script or English. Personal values are masked as [ACCOUNT_n], [PHONE_n], [NAME_n].\n"
    "Return JSON only, with exactly these keys:\n"
    '  "account_number": the [ACCOUNT_n] placeholder of the customer account, or null when the message has none\n'
    f'  "district": one of {list(District.__args__)}, or null when the message names no district\n'
    f'  "complaint_type": one of {list(ComplaintType.__args__)}\n'
    "Districts in Arabic: الغدير ghadeer, المنهل manhal, الشرفة shurfa, الورود wurood, الياسمين yasmeen, القلعة qalaa, "
    "الربوة rabwa, النسيم naseem. Never invent a value.")
FAILURES = []


def extract(item: dict) -> dict | None:
    masked, mapping = mask(item["text"])
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Message: " + masked}]
    # BEGIN SOLUTION
    for attempt in (1, 2):
        reply = G.chat(messages, json_mode=True, max_tokens=150)["text"]
        try:
            fields = validate(reply, mapping).model_dump()
            fields["account_number"] = unmask(fields["account_number"], mapping)
            return fields
        except (ValidationError, ValueError) as err:
            error = "; ".join(e["msg"] for e in err.errors()) if isinstance(err, ValidationError) else str(err)
            FAILURES.append({"msg_id": item["msg_id"], "attempt": attempt, "error": error[:200]})
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": "Your reply failed validation: " + error + ". Return the corrected JSON only."}]
    return None
    # END SOLUTION


# %%
def run_extraction():
    FAILURES.clear()
    preds = [extract(it) for it in ITEMS]
    return {"predictions": preds, "failures": list(FAILURES)}


run = G.attempt(run_extraction, "d22_run")
if len(run["predictions"]) != len(ITEMS):  # the reference result covers all 200 messages
    ITEMS = G.jsonl("extraction_200.jsonl")
PREDS, FAILURES = run["predictions"], run["failures"]
first, second = sum(f["attempt"] == 1 for f in FAILURES), sum(f["attempt"] == 2 for f in FAILURES)
print(f"{len(ITEMS)} messages | failed the first validation: {first} | still failing after one repair: {second} |",
      "stand-in (scripted, not a model)" if G.standin() else G.CHAT_MODEL)
print(pd.Series([f["error"][:70] for f in FAILURES]).value_counts().head(5).to_string())
out = G.lab_dir("D2_2")
(out / "validation_failures.jsonl").write_text("\n".join(json.dumps(f, ensure_ascii=False) for f in FAILURES) + "\n", encoding="utf-8")
assert len(PREDS) == len(ITEMS) and sum(p is None for p in PREDS) == second

# %% [markdown]
# 🔍 Read the failure log by kind. Which failures did the repair fix, and which came back unchanged? The log
# is part of this session's evidence: it shows what the validators stopped.

# %% [markdown]
# ## 4 · Field precision and recall against a regex baseline (6')
#
# For one field: **precision** = correct values / values the extractor returned; **recall** = correct values /
# values the gold data holds. A `null` is not a value: returning `null` where the gold has a district costs
# recall; returning a district where the gold has none costs precision.
#
# ✏️ **Exercise** · Write `field_scores(preds, golds)`: for each of the three fields a dictionary with
# `precision` and `recall`, rounded to 3 decimals. A prediction of `None` for the whole message (a logged
# failure) counts as `null` in every field.

# %%
FIELDS = ["account_number", "district", "complaint_type"]


def field_scores(preds: list, golds: list) -> dict:
    # BEGIN SOLUTION
    out = {}
    for f in FIELDS:
        p = [(pr or {}).get(f) for pr in preds]
        g = [go[f] for go in golds]
        correct = sum(a is not None and a == b for a, b in zip(p, g))
        returned, held = sum(a is not None for a in p), sum(b is not None for b in g)
        out[f] = {"precision": round(correct / returned, 3) if returned else 0.0, "recall": round(correct / held, 3) if held else 0.0}
    return out
    # END SOLUTION


def regex_extract(text: str) -> dict:
    """The baseline: the first run of ten digits, the first district name found, a keyword list for the type."""
    number = re.search(r"\d{10}", text)
    district = next((key for key, names in G.district_names().items() if any(G.normalise(n) in G.normalise(text) for n in names)), None)
    return {"account_number": number.group(0) if number else None, "district": district, "complaint_type": G.keyword_type(text)}


# %%
GOLDS = [it["gold"] for it in ITEMS]
scores = G.attempt(lambda: {"regex baseline": field_scores([regex_extract(it["text"]) for it in ITEMS], GOLDS),
                            "model, masked and validated": field_scores(PREDS, GOLDS)}, "d22_scores")
table = pd.DataFrame({(who, f): s[f] for who, s in scores.items() for f in FIELDS}).T
print(table.to_string())
table.to_csv(out / "field_scores.csv")
wrong = [(it["text"], regex_extract(it["text"])["account_number"], it["gold"]["account_number"]) for it in ITEMS
         if regex_extract(it["text"])["account_number"] != it["gold"]["account_number"]]
print(f"\nregex baseline, wrong account number in {len(wrong)} messages. Two of them:")
for text, got, gold in wrong[:2]:
    print("  ", text[:110], "\n     got", got, "| gold", gold)

# %% [markdown]
# 🔍 Two planted problems explain the baseline's account-number errors: a number in Arabic-Indic digits is
# returned as written and does not equal the gold value, and a message with a phone number and no account
# number gives the phone number. Which line of `regex_extract` would you change for each? And what does the
# district column tell you about the word المنهل?

# %% [markdown]
# ## ✱ Stretch · Add two examples per variety to the prompt and score again
#
# Take eight messages from the training split (two per variety), mask them, and add them to the system
# prompt as examples with their correct JSON. Does the complaint type improve for Arabizi? What did the
# longer prompt cost per message, in tokens?

# %%
# BEGIN SOLUTION
train = G.messages("train")
shots = pd.concat([train[train["variety"] == v].head(2) for v in G.VARIETIES])
examples = "\n".join(f"Message: {mask(r.text)[0]}\nJSON: " + json.dumps({"account_number": "[ACCOUNT_1]" if r.account_number else None,
                     "district": r.district, "complaint_type": r.complaint_type}, ensure_ascii=False) for r in shots.itertuples())
print(len(examples) // 4, "more tokens per call, roughly")
# END SOLUTION

# %% [markdown]
# ## Checkpoint
#
# 1. The model returns valid JSON every time. What have you still not measured?
# 2. Why does the validator need the mapping of this message, and not only a pattern for placeholders?
# 3. A message has no account number and the model returns one. Which number in the table shows it?
# 4. Classify each field this lab handles (name, phone number, account number, district, complaint type) with
#    Sawaqi's rule R-10 and the levels of the national Data Classification Policy, read at its official
#    address in the course sources. Which fields may reach the hosted model unmasked?
#
# <details><summary>Answers</summary>
#
# 1. Whether the values are right. Valid JSON with the wrong district passes every parser. Field precision
#    and recall against gold fields measure it.
# 2. A pattern accepts `[ACCOUNT_3]` in a message that has one account. Only the mapping says which
#    placeholders exist in this message.
# 3. Precision of the account number: a returned value where the gold holds none is a false positive.
# 4. R-10 places customer records at the Restricted level. The name, the phone number and the account number
#    identify a customer: they are masked. The district and the complaint type on their own do not, and the
#    model needs them, so they leave unmasked. Check the levels and their definitions in the policy itself
#    before you classify real data.
#
# </details>
#
# **Carry forward.** `artifacts/D2_2/field_scores.csv` and `validation_failures.jsonl`. `mask` and the schema
# come back in Lab 3.4's service.
