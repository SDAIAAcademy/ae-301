# %% [markdown]
# # Lab 1.1 · Three tokenisers, four varieties
#
# Day 1 · Session 1 (Tokenisation) · 25 min. Sawaqi's customers write in Modern Standard Arabic, Gulf dialect,
# Arabic in Latin script (Arabizi) and English. Before any model reads a message, a tokeniser cuts it into
# pieces, and the number of pieces is what you pay for. You count tokens for 50 parallel messages with three
# tokenisers, price the difference, write a normaliser and trace two failures to their cause.
#
# **Reference state.** `data/parallel_50.csv`: 50 messages, each written in all four varieties with the same
# facts. Three tokeniser files from the Hugging Face Hub (vocabulary files of a few megabytes; no model is
# downloaded). Nothing from another lab.
#
# **Tasks**
# - [ ] 1 · Load three tokenisers and split one message (4')
# - [ ] 2 · Tokens per message: 50 messages, four varieties (7')
# - [ ] 3 · Price the difference per thousand messages (5')
# - [ ] 4 · Normalise, then trace two failures to their cause (9')
# - [ ] ✱ Stretch · Train a small tokeniser on the Sawaqi messages and count again

# %%
#@setup
G.banner("Lab 1.1")

# %% [markdown]
# ## 1 · Load three tokenisers and split one message (4')
#
# | Name | Kind | Trained on |
# |---|---|---|
# | `gpt2` | byte-level BPE, 50,257 entries | English web text |
# | `qwen3` | byte-level BPE, about 151,000 entries | many languages |
# | `camelbert` | WordPiece, 30,000 entries | Arabic: MSA, dialects and classical |
#
# A byte-level tokeniser never fails: any text is bytes. It shows its pieces as bytes, so Arabic pieces look
# garbled when printed; decode them to read them. WordPiece marks a piece that continues a word with `##`
# and answers `[UNK]` for a word it cannot build.

# %%
import pandas as pd

PAR = pd.read_csv(G.DATA / "parallel_50.csv")
TOK = G.hub_tokenisers()
row = PAR.iloc[0]
for variety in G.VARIETIES:
    print(f"\n{variety}: {row[variety]}")
    for name, tok in TOK.items():
        enc = tok.encode(row[variety], add_special_tokens=False)
        pieces = [tok.decode([i]) for i in enc.ids] if name != "camelbert" else enc.tokens
        print(f"  {name:10s}{len(enc.ids):3d} tokens  {pieces[:12]}")

# %% [markdown]
# 🔍 Read the GPT-2 line for the MSA message: most of its pieces are single bytes, shown as `�` because half an
# Arabic letter is not a character. Which tokeniser has the fewest pieces for English, and which for Arabic?

# %% [markdown]
# ## 2 · Tokens per message: 50 messages, four varieties (7')
#
# ✏️ **Exercise** · Write `token_counts(par, toks)`. For every tokeniser and every variety, return one dictionary
# with `tokeniser`, `variety`, `tokens_per_message` (the mean over the 50 messages) and `tokens_per_word`
# (all tokens divided by all whitespace-separated words), both rounded to 2 decimals.
# `G.n_tokens(tokeniser, text)` counts the tokens of one text.

# %%
def token_counts(par: pd.DataFrame, toks: dict) -> list[dict]:
    # BEGIN SOLUTION
    rows = []
    for name, tok in toks.items():
        for v in G.VARIETIES:
            n = [G.n_tokens(tok, t) for t in par[v]]
            words = sum(len(t.split()) for t in par[v])
            rows.append({"tokeniser": name, "variety": v, "tokens_per_message": round(sum(n) / len(n), 2),
                         "tokens_per_word": round(sum(n) / words, 2)})
    return rows
    # END SOLUTION


# %%
# Check, with a safety net: the cells below run even when the exercise is not done or a tokeniser did not download
counts = G.attempt(lambda: token_counts(PAR, TOK), "d11_counts") if len(TOK) == 3 else G.reference("d11_counts")
COUNTS = pd.DataFrame(counts)
print(COUNTS.pivot(index="tokeniser", columns="variety", values="tokens_per_message")[G.VARIETIES].to_string())
print()
print(COUNTS.pivot(index="tokeniser", columns="variety", values="tokens_per_word")[G.VARIETIES].to_string())
per_msg = COUNTS.set_index(["tokeniser", "variety"])["tokens_per_message"]
G.check(per_msg["gpt2", "msa"] > 3 * per_msg["gpt2", "en"], "GPT-2 spends more than three times the tokens on the MSA version")
G.check(per_msg["camelbert", "msa"] < per_msg["qwen3", "msa"], "the Arabic tokeniser is the shortest on MSA")
G.check(per_msg["camelbert", "arabizi"] > per_msg["camelbert", "msa"], "and it pays for Arabizi, which it never saw as words")

# %% [markdown]
# 🔍 The same complaint with the same facts costs a different number of tokens in each variety, and the order
# changes with the tokeniser. No tokeniser here is good at Arabizi: nobody trained one on it.

# %% [markdown]
# ## 3 · Price the difference per thousand messages (5')
#
# A hosted model charges per token. The price below is the published input price of the paid endpoint of the
# course's chat model, read on the date in `data/reference/prices.json`; set your own provider's price instead.
# Each routing call also sends the instructions: 180 tokens before the message.
#
# ✏️ **Exercise** · Write `cost_per_1000(tokens_per_message, usd_per_million, prompt_tokens)`: the cost in USD
# of sending 1,000 messages, each with the instructions in front.

# %%
PRICES = G.reference("prices")
USD_PER_MILLION = PRICES["models"][PRICES["priced_model"]]["prompt"] * 1e6
PROMPT_TOKENS = 180
print(f"price: {USD_PER_MILLION:.2f} USD per million input tokens ({PRICES['priced_model']}, read {PRICES['read_on']})")


def cost_per_1000(tokens_per_message: float, usd_per_million: float, prompt_tokens: int = 0) -> float:
    # BEGIN SOLUTION
    return round((tokens_per_message + prompt_tokens) * 1000 * usd_per_million / 1e6, 4)
    # END SOLUTION


# %%
try:
    cost_per_1000(10, 1.0)
except NotImplementedError:
    def cost_per_1000(tokens_per_message, usd_per_million, prompt_tokens=0):
        return round((tokens_per_message + prompt_tokens) * 1000 * usd_per_million / 1e6, 4)

assert cost_per_1000(20, 0.5) == 0.01 and cost_per_1000(20, 0.5, 180) == 0.1
qwen = per_msg["qwen3"]
price = pd.DataFrame({"tokens_per_message": qwen,
                      "usd_per_1000_message_only": [cost_per_1000(n, USD_PER_MILLION) for n in qwen],
                      "usd_per_1000_with_prompt": [cost_per_1000(n, USD_PER_MILLION, PROMPT_TOKENS) for n in qwen]})
price["vs_english"] = (price["usd_per_1000_message_only"] / price.loc["en", "usd_per_1000_message_only"]).round(2)
print(price.to_string())

# %% [markdown]
# 🔍 Two readings. The message itself costs more in Arabic and Arabizi than in English. With the instructions
# in front, the difference almost disappears: the prompt is most of the bill. Which of the two would you cut?

# %% [markdown]
# ## 4 · Normalise, then trace two failures to their cause (9')
#
# Real messages carry spelling that means nothing to the reader and everything to a tokeniser: diacritics
# (العَدّاد), the stretching mark tatweel (العـداد), alef with or without hamza (أرجو, ارجو), ta marbuta
# written as ha (الفاتورة, الفاتوره), alef maqsura for ya (في, فى), and Arabic-Indic digits (٣٠).
#
# ✏️ **Exercise** · Write `normalise(text)`: remove diacritics (U+064B to U+0652, U+0670) and tatweel (U+0640);
# turn أ إ آ into ا; ة into ه; ى into ي; Arabic-Indic digits into Western digits; lower-case Latin letters.

# %%
import re


def normalise(text: str) -> str:
    # BEGIN SOLUTION
    text = re.sub("[ً-ْٰـ]", "", text)
    text = re.sub("[أإآ]", "ا", text).replace("ة", "ه").replace("ى", "ي")
    return text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")).lower()
    # END SOLUTION


# %%
try:
    normalise("أ")
except NotImplementedError:
    normalise = G.normalise  # the reference version, so the cells below run

assert normalise("العَدّاد") == "العداد" and normalise("العـداد") == "العداد", "diacritics and tatweel"
assert normalise("أرجو إغلاق الفاتورة فى") == "ارجو اغلاق الفاتوره في", "alef, ta marbuta, alef maqsura"
assert normalise("٣٠٦١٠٧٤٦١٠ SAR") == "3061074610 sar", "digits and case"
print("✅ normalise passes its three tests")

# %% [markdown]
# Now the two failures. For each, the table shows the pieces before and after normalising.

# %%
cases = [("a diacritic and a shadda", "العَدّاد واقف"), ("tatweel", "العـداد واقف"),
         ("Arabic-Indic digits", "رقم الحساب ٣٠٦١٠٧٤٦١٠")]
rows = []
for why, text in cases:
    for name, tok in TOK.items():
        rows.append({"case": why, "tokeniser": name, "tokens": G.n_tokens(tok, text), "after": G.n_tokens(tok, normalise(text)),
                     "unknown": tok.encode(text, add_special_tokens=False).tokens.count("[UNK]")})
trace = pd.DataFrame(rows)
print(trace.to_string(index=False))

# On real messages: how many word forms does the archive hold before and after?
arabic = G.messages()
arabic = arabic[arabic["variety"].isin(["msa", "gulf"])]["text"]
before = {w for t in arabic for w in t.split()}
after = {w for t in arabic for w in normalise(t).split()}
print(f"\nword forms in {len(arabic)} Arabic-script messages: {len(before)} before, {len(after)} after normalising")
if "camelbert" in TOK:
    unk = lambda texts: sum(TOK["camelbert"].encode(t, add_special_tokens=False).tokens.count("[UNK]") for t in texts)
    print(f"[UNK] pieces from the Arabic tokeniser: {unk(arabic)} before, {unk(map(normalise, arabic))} after")

# %% [markdown]
# 🔍 **Failure 1.** One diacritic turns a known word into `[UNK]` for the Arabic tokeniser, and into a run of
# single letters for the multilingual one. Cause: the tokeniser's training text had no diacritics, so the
# vocabulary has no piece that contains one. **Failure 2.** An account number in Arabic-Indic digits costs
# twice the tokens of the same number in Western digits. Cause: each of those digits is two bytes, and the
# byte-level vocabulary has no merged piece for them.
#
# ✏️ **Your note (the evidence of this session).** In `artifacts/D1_1/note.md`, the cell below writes the token
# counts per variety. Add two sentences in your own words: each failure and its cause.

# %%
note = G.lab_dir("D1_1") / "note.md"
table = COUNTS.pivot(index="tokeniser", columns="variety", values="tokens_per_message")[G.VARIETIES]
note.write_text("\n".join(["# Tokenisers at Sawaqi: what a message costs", "", "Tokens per message, 50 parallel messages:", "",
                           "```", table.to_string(), "```", "", "Two failures, tokens before and after normalising:", "",
                           "```", trace.to_string(index=False), "```", "", "Failure 1 (cause): ...", "", "Failure 2 (cause): ...", ""]),
                encoding="utf-8")
print("wrote", note.relative_to(G.LABS))

# %% [markdown]
# ## ✱ Stretch · Train a small tokeniser on the Sawaqi messages and count again
#
# The course trained a byte-pair tokeniser with 2,000 entries on Sawaqi's own messages
# (`models/ghayth-bpe.json`; `_tools/pretrain_tiny.py` shows how). Count the 50 parallel messages with it. Why
# is it shorter on Arabizi than all three? What would it do with a message about something Sawaqi never saw?

# %%
# BEGIN SOLUTION
import tinytf as T

own = T.load_tokenizer()
print({v: round(sum(G.n_tokens(own, t) for t in PAR[v]) / len(PAR), 2) for v in G.VARIETIES})
print(own.encode("the invoice of my restaurant").tokens)
# END SOLUTION

# %% [markdown]
# ## Checkpoint
#
# 1. Why does GPT-2 never answer `[UNK]`, and what does it do instead with Arabic?
# 2. The instructions are 180 tokens and the message 15. Where would you look first to cut the bill?
# 3. Normalising ة to ه merges two different words in some cases. When is that a price worth paying?
#
# <details><summary>Answers</summary>
#
# 1. It works on bytes, so any text can be written with its 256 byte pieces. Arabic, which it rarely saw, falls
#    back to those single bytes: two per letter.
# 2. The instructions: they are sent with every message. Shorten them or use a provider that caches a prefix.
# 3. For search and classification, where matching the same word in two spellings matters more than the rare
#    pair it merges. Not for text you show to a customer: keep the original for display.
#
# </details>
#
# **Carry forward.** `normalise` is `G.normalise` in every later lab. Your note is in `artifacts/D1_1/note.md`.
