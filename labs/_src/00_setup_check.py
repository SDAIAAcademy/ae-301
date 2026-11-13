# %% [markdown]
# # Lab 0 · Setup check
#
# Run this notebook **before Day 1**. It takes about two minutes. It checks the Python packages, reads the
# course data, loads the course's two small models, fetches the three tokeniser files of Lab 1.1 and calls
# the hosted model once. Nothing downloads a language model: the only downloads are three tokeniser files
# of a few megabytes.
#
# Locally: `cd labs && uv sync && uv run jupyter lab`. In Colab: run the first cell; add
# `OPENROUTER_API_KEY` in the Secrets panel and allow notebook access.

# %%
#@setup
G.banner("Setup check")

# %% [markdown]
# ## 1 · Packages

# %%
import importlib
import platform

print("Python", platform.python_version(), "|", platform.system())
missing = []
for name in ["numpy", "pandas", "sklearn", "torch", "tokenizers", "huggingface_hub", "openai", "pydantic", "fastapi", "httpx", "matplotlib"]:
    try:
        mod = importlib.import_module(name)
        print(f"  ✅ {name} {getattr(mod, '__version__', '')}")
    except ImportError:
        missing.append(name)
        print(f"  ❌ {name} is missing")
assert not missing, f"install the missing packages: {missing} (locally: uv sync; in Colab: run the first cell again)"

# %% [markdown]
# ## 2 · The case data and the course's models

# %%
import tinytf as T

m = G.messages()
print(len(m), "messages |", m["split"].value_counts().to_dict())
print(m["variety"].value_counts().to_dict())
assert len(m) == 2000 and len(G.rules()) == 10 and len(G.jsonl("queries.jsonl")) == 40
tok = T.load_tokenizer()
for name in ["ghayth-enc-tiny.pt", "ghayth-dec-tiny.pt"]:
    print(f"  ✅ {name}: {T.count_params(T.load(G.MODELS / name)):,} parameters")
router, labels = T.load_classifier(G.MODELS / "ghayth-enc-routing-full.pt")
print("  ✅ the supplied fine-tuned encoder routes to", len(labels), "departments")

# %% [markdown]
# ## 3 · Tokeniser files from the Hub (Lab 1.1)

# %%
toks = G.hub_tokenisers()
for name, t in toks.items():
    print(f"  ✅ {name}: {t.get_vocab_size():,} entries")
if len(toks) < 3:
    print("  A tokeniser file could not be fetched (a blocked network?). Lab 1.1 then uses the reference counts.")

# %% [markdown]
# ## 4 · One call to the hosted model (Day 2) and one embedding (Lab 1.2)
#
# Without a working key the labs still run: a scripted stand-in answers and every lab says so in its first
# line. Its numbers are not a model's, so get a key before Day 2. Free endpoints have a daily request limit;
# `ANLP_QUICK=1` keeps every lab inside it.

# %%
r = G.chat([{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=5)
print("chat:", "STAND-IN (no working key)" if r["standin"] else f"{r['model']} answered {r['text']!r} in {r['latency_s']} s")
v = G.embed(["الماي مقطوع عندنا من الصبح"])
print("embedding:", "STAND-IN (no working key)" if G.standin() else f"{G.EMBED_MODEL}, {v.shape[1]} dimensions")
print("\nSetup check finished.", "You are ready for all three days." if not G.standin() else "Days 1 and 3 will run as they are; fix the key before Day 2.")
