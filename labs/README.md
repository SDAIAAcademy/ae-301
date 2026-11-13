# Applied Natural Language Processing Engineering: labs

Hands-on notebooks for the SDAIA Academy course **AE-301 Applied Natural Language Processing Engineering (الهندسة التطبيقية لمعالجة اللغات الطبيعية)**. Every lab builds a piece of **Ghayth (غيث)**, the language layer of **Sawaqi Water Company**, a fictional regional water utility whose customers write in Modern Standard Arabic, Gulf dialect, Arabic in Latin script (Arabizi) and English. Each lab starts from supplied files, so none depends on an earlier one.

**Nothing downloads a language model.** Generative and embedding models are free hosted open-weight models reached over an API. The models you open, fine-tune and adapt are the course's own small encoder and decoder (about half a million parameters each, in `models/`), which train on a laptop processor in under a minute. A laptop CPU or a Colab CPU runtime is enough for all three days.

## Quick start

**1 · Get a free OpenRouter key** at <https://openrouter.ai/keys>. Free models (ids ending in `:free`) allow 20 requests a minute and 50 a day; an account that has bought at least 10 credits gets 1,000 free-model requests a day, and free models do not spend the credits. Identical requests are replayed from `labs/cache/`, so running a cell again costs nothing. The helper spaces calls to a free model by about three seconds and waits when the provider answers with a rate limit. **Plan Day 2 before the cohort:** its four labs make about 390 requests (about 125 with `ANLP_QUICK=1`), more than a fresh free key allows in a day even in quick mode. Each participant needs a key on an account that has bought at least 10 credits, or a table shares one such key and its cache; Day 1 (about 22 requests) fits a fresh free key.

**2a · Google Colab**
1. Open a notebook from the [course page](https://sdaiaacademy.github.io/applied-nlp/), or directly: `https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/<notebook>.ipynb`.
2. In the 🔑 **Secrets** panel add `OPENROUTER_API_KEY` and turn on notebook access.
3. Run the first cell. It clones this repository and installs `requirements-colab.txt`.

**2b · Local**
```bash
git clone https://github.com/SDAIAAcademy/applied-nlp
cd applied-nlp/labs
cp .env.example .env           # then paste your key into .env
uv sync                        # creates .venv from pyproject.toml (uv.lock pins every version)
uv run jupyter lab
```
Run `00_setup_check.ipynb` once before Day 1.

## Models and settings

| Setting (`.env`) | Default | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | none | or `LLM_API_KEY` with `LLM_BASE_URL` for any provider that serves the OpenAI API shape |
| `ANLP_MODEL` | `google/gemma-4-31b-it:free` | the chat model of Day 2; any free open-weight chat model |
| `ANLP_SMALL_MODEL` | `liquid/lfm-3-1.7b:free` | the second model of Lab 2.4 |
| `ANLP_EMBED_MODEL` | `nvidia/nemotron-3.5-embed-1b:free` | the embedding model of Lab 1.2; any hosted open multilingual embedding model |
| `ANLP_FALLBACKS` | `google/gemma-4-26b-a4b-it:free` | tried in order when a free model is busy |
| `ANLP_QUICK` | `0` | `1` shrinks samples and epochs: a free daily quota, a slow laptop, a quick check |
| `ANLP_CACHE` | `1` | `0` sends every request to the model again |
| `ANLP_STANDIN` | `0` | `1` never calls a model |

Free models change often. If a default has disappeared or is busy, pick another free open-weight model from the provider's list and set it here. Model names are examples, not endorsements.

**The stand-in.** When no key works, a scripted stand-in answers in place of the hosted model, and the first line of every lab says so. It is a keyword script, not a model: it lets every cell run, and its scores, latencies and token counts say nothing about a model. The reference results in `data/reference/` for Labs 1.2, 2.1, 2.2, 2.3 and 2.4 were produced with it, because the course was built without access to a hosted model; replace them with your own runs.

## Notebooks

| Day | Session | Notebook | Needs a hosted model |
|---|---|---|---|
| 0 | Setup | [`00_setup_check.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/00_setup_check.ipynb) | one call |
| 1 | Tokenisation | [`D1_1_tokenisers.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/D1_1_tokenisers.ipynb) | no (three tokeniser files from the Hub) |
| 1 | Embeddings and semantic search | [`D1_2_semantic_search.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/D1_2_semantic_search.ipynb) | embeddings, 21 requests |
| 1 | Inside the transformer | [`D1_3_attention.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/D1_3_attention.ipynb) | no |
| 1 | Decision models | [`D1_4_baseline.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/D1_4_baseline.ipynb) | no · **AI assistants off** |
| 2 | Generative language models | [`D2_1_sampling.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/D2_1_sampling.ipynb) | chat, 60 requests (24 in quick mode) |
| 2 | Prompting and structured outputs | [`D2_2_extraction.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/D2_2_extraction.ipynb) | chat, about 230 requests (about 45 in quick mode) |
| 2 | Retrieval-augmented generation | [`D2_3_rag.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/D2_3_rag.ipynb) | chat, 16 requests |
| 2 | Model selection and evaluation | [`D2_4_model_selection.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/D2_4_model_selection.ipynb) | chat, 80 requests (40 in quick mode) |
| 3 | Fine-tuning data | [`D3_1_splits.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/D3_1_splits.ipynb) | no |
| 3 | Fine-tuning an encoder | [`D3_2_finetune_encoder.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/D3_2_finetune_encoder.ipynb) | no |
| 3 | Low-rank and quantised adapters | [`D3_3_lora.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/D3_3_lora.ipynb) | no |
| 3 | Evaluate and deploy | [`D3_4_service.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/D3_4_service.ipynb) | no |

Exercises are marked **✏️ Exercise** and raise `NotImplementedError` until you fill them in. The cell after each exercise has a safety net: when the exercise is not done, it takes a reference result, so the rest of the notebook still runs. Complete versions are in `solutions/`. Cells marked **✱ Stretch** are optional. Each lab writes its outputs to `artifacts/<lab>/`; a later lab that compares with an earlier one reads your own output when it exists and the supplied copy in `data/reference/` otherwise.

**AI policy.** Assistants are off in the supervised checks and in Lab 1.4. In the other labs an assistant may hint; it does not write your training or evaluation code. In the capstone assistants are allowed in full, and the work is defended live.

## The case study: Sawaqi Water Company

Synthetic and deterministic (`prepare_data.py`, standard library only, fixed seed); no real person, account or phone number.

| Data | File | Contents |
|---|---|---|
| Customers | `data/customers.csv` | 700 customers with invented names, phone numbers (`05` and eight digits) and account numbers (`30` and eight digits) |
| Messages | `data/messages.csv` | 2,000 messages: MSA 33%, Gulf 34%, English 21%, Arabizi 12%; department (8), complaint type (16), gold account number and district |
| Split | `data/split.csv` | by customer: `train` (402 messages the contact centre labelled), `val` (301), `test` (403), `pool` (894: the rest of the archive, used for search, never for its routing labels) |
| Parallel messages | `data/parallel_50.csv` | 50 messages written in all four varieties with the same facts (Lab 1.1) |
| Search queries | `data/queries.jsonl` | 40 queries, 10 per variety, with their relevant messages (Lab 1.2) |
| Gold fields | `data/extraction_200.jsonl` | 200 test messages with account number, district and complaint type (Lab 2.2) |
| Rule book | `data/rules/R-01 … R-10` | the tariff rules, each in English and Arabic; `data/rule_questions.jsonl` has 16 questions, 4 of which the rules do not answer (Lab 2.3) |
| Reply tasks | `data/reply_tasks.jsonl` | ten messages with the facts a reply may use (Lab 2.1) |
| Smoke test | `data/smoke_test.jsonl` | 20 test messages and 4 edge cases that must go to a person (Lab 3.4) |
| Reference results | `data/reference/` | results of the course's solution runs, for the safety nets and for labs that compare with an earlier lab; the published prices read on the date inside `prices.json` |
| Models | `models/` | the course tokeniser (2,000 entries), encoder and decoder, and the encoder fine-tuned for routing; `_tools/pretrain_tiny.py` trained them |

The planted problems and the expected numbers of every lab are in `solutions/answer_key.md`, for the trainer.

## Helpers

| Module | What it holds |
|---|---|
| `ghayth.py` | paths, quick mode, loading the data, `normalise`, scores by variety, latency percentiles, one cached call to a hosted model (`chat`, `embed`), published prices, the stand-in, the safety nets (`attempt`, `reference`, `supplied`) and the Day 1 baseline (`baseline`) |
| `tinytf.py` | the course's small transformer in about 150 lines: attention with a causal switch, the encoder and decoder, a classifier, and reference versions of Lab 3.3's adapter code |
| `capstone/` | the starter kit of the capstone: the service as one file, an evaluation script that writes timestamped logs, the assessor's batch format and a README template |

Delete `artifacts/` and `cache/` before running the course's `academy check`, which reports them as files that are not part of the course.

**Windows and long folder paths.** If an import fails with "The filename or extension is too long", keep the virtual environment in a short folder: set `UV_PROJECT_ENVIRONMENT` to a short path (a folder at the root of your drive) before `uv sync`.
