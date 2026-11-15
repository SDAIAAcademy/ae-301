<!-- academy:title -->
# Applied Natural Language Processing Engineering

الهندسة التطبيقية لمعالجة اللغات الطبيعية

AI Engineering Track · SDAIA Academy · AE-301 · Level 300

Route, search and extract customer messages in four varieties of Arabic and English, and choose the model on measured quality, latency and cost

Participants build the language layer of a customer-service system for a fictional regional water utility whose messages arrive in Modern Standard Arabic, Gulf dialect, Arabic written in Latin script and English. They trace text through tokenisers, embeddings and attention, build semantic search, route messages and extract account fields three ways (a classical baseline, a prompted generative model and a fine-tuned encoder), adapt a model with low-rank adapters in free notebook memory, and ship the chosen model as a small service with a model card. Every choice rests on scores by dialect, latency and cost.

**Format:** Intensive: 3 teaching days of four sessions and a capstone day, with guided pre-work and a tracked follow-up task

**Course page:** https://sdaiaacademy.github.io/ae-301/
<!-- /academy:title -->

The whole course follows one case study: **Sawaqi Water Company**, a fictional regional water utility, and **Ghayth
(غيث)**, the language layer participants build for its customer care. Customers write in four varieties: Modern
Standard Arabic, Gulf dialect, Arabic in Latin script (Arabizi) and English. The 2,000 messages, 700 customers, the
tariff rule book, the queries and the gold fields are synthetic and generated with a fixed seed by
`labs/prepare_data.py`; every name, account number and phone number in them is invented.

Every lab runs on a laptop CPU with [uv](https://docs.astral.sh/uv/) or on Google Colab. Day 1 (except one step) and
Day 3 need no model service at all; Day 2 calls free hosted open-weight models (OpenRouter by default). Nothing
downloads a language model: the encoder and decoder that the labs open and fine-tune are two small models trained for
this course and shipped in `labs/models/` (about 2 MB each). See [labs/README.md](labs/README.md).

One result to know before teaching: on this data the Day 1 baseline (character and word n-grams with a linear model,
macro F1 0.85) beats the fine-tuned course encoder (0.63) and the low-rank adapter (0.50), and no approach is good at
Arabizi. The decks say so and the service ships the baseline with a hand-off rule. The course teaches choosing on
evidence; it does not promise that the neural model wins.

<!-- academy:units -->
## Units

| Unit | Title | Sessions | Slides |
|---|---|---|---|
| Day 1 | Represent | Tokenisation; Embeddings and semantic search; Inside the transformer; Decision models | [`slides/day1.html`](slides/day1.html) |
| Day 2 | Generate | Generative language models; Prompting and structured outputs; Retrieval-augmented generation; Model selection and evaluation | [`slides/day2.html`](slides/day2.html) |
| Day 3 | Adapt | Fine-tuning data; Fine-tuning an encoder; Low-rank and quantised adapters; Evaluate and deploy | [`slides/day3.html`](slides/day3.html) |
| Capstone | Capstone: a routing and extraction service and its defence |  | [`slides/day4.html`](slides/day4.html) |
<!-- /academy:units -->

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | Three tokenisers, four varieties | [`D1_1_tokenisers.ipynb`](labs/D1_1_tokenisers.ipynb) |
| 1.2 | Day 1 · Session 2 | Semantic search over 2,000 messages | [`D1_2_semantic_search.ipynb`](labs/D1_2_semantic_search.ipynb) |
| 1.3 | Day 1 · Session 3 | Attention in an encoder and a decoder | [`D1_3_attention.ipynb`](labs/D1_3_attention.ipynb) |
| 1.4 | Day 1 · Session 4 | A routing baseline, without AI | [`D1_4_baseline.ipynb`](labs/D1_4_baseline.ipynb) |
| 2.1 | Day 2 · Session 1 | Sampling, consistency and invented details | [`D2_1_sampling.ipynb`](labs/D2_1_sampling.ipynb) |
| 2.2 | Day 2 · Session 2 | Masked, validated extraction | [`D2_2_extraction.ipynb`](labs/D2_2_extraction.ipynb) |
| 2.3 | Day 2 · Session 3 | Grounded replies from the rule book | [`D2_3_rag.ipynb`](labs/D2_3_rag.ipynb) |
| 2.4 | Day 2 · Session 4 | Two open-weight models against the baseline | [`D2_4_model_selection.ipynb`](labs/D2_4_model_selection.ipynb) |
| 3.1 | Day 3 · Session 1 | A split without leakage | [`D3_1_splits.ipynb`](labs/D3_1_splits.ipynb) |
| 3.2 | Day 3 · Session 2 | Fine-tune the course encoder for routing | [`D3_2_finetune_encoder.ipynb`](labs/D3_2_finetune_encoder.ipynb) |
| 3.3 | Day 3 · Session 3 | A low-rank adapter in a memory budget | [`D3_3_lora.ipynb`](labs/D3_3_lora.ipynb) |
| 3.4 | Day 3 · Session 4 | A small service with a model card | [`D3_4_service.ipynb`](labs/D3_4_service.ipynb) |
<!-- /academy:labs -->

## Capstone

A routing and extraction service for Sawaqi, or for a comparable text task with synthetic data, built on the starter in
[labs/capstone/](labs/capstone/). You publish it in a public repository whose README reports:

- the three-way comparison (classical baseline, prompted generative model, fine-tuned encoder) by language variety,
  with CPU latency and cost per thousand messages;
- field-level precision and recall of the masked, validated extraction;
- the full fine-tuning run and the adapter run, with trainable-parameter counts and logs;
- the model card, with results by variety and the hand-off rule.

Then you defend it for eight minutes and answer questions for four: you re-score your served model on a batch the
assessor supplies and explain two of its errors.

## Repository

```
course.json   the course manifest: outcomes, days, sessions, labs, capstone, tools, sources
slides/       one deck per day; the engine builds them into the course site
assessments/  bank.json: 27 practice questions by course outcome
labs/         notebooks and their solutions (built from _src/ by _tools/py2nb.py), the data and its generator
              (prepare_data.py, data/), the helper module (ghayth.py), the small transformer (tinytf.py, models/),
              the capstone kit (capstone/)
```

Build and check with the Academy engine: `node academy.mjs check`, `node academy.mjs build`.

<!-- academy:tools -->
## Tools

Each line is a capability; the examples are interchangeable.

- **Model library and training framework:** Hugging Face Transformers, PyTorch
- **Embeddings and vector index:** Sentence Transformers, FAISS, LanceDB
- **Baselines and metrics:** scikit-learn, seqeval
- **Parameter-efficient fine-tuning:** PEFT, a hand-written low-rank adapter layer
- **Arabic text processing:** CAMeL Tools, PyArabic
- **Language model access:** a locally served open-weight model, a free hosted open-weight model, any commercial API
- **Service, validation and demo:** FastAPI with Pydantic, Gradio, Streamlit
<!-- /academy:tools -->

<!-- academy:sources -->
## Sources

### Tokenisers, transformers and encoders

- [Vaswani et al.: Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Sennrich, Haddow and Birch: Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- [Devlin et al.: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Inoue et al.: The Interplay of Variant, Size, and Task Type in Arabic Pre-trained Language Models (CAMeLBERT)](https://arxiv.org/abs/2103.06678)
- [Hugging Face: CAMeL-Lab/bert-base-arabic-camelbert-mix (tokeniser used in Lab 1.1)](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix)
- [Hugging Face: openai-community/gpt2 (tokeniser used in Lab 1.1)](https://huggingface.co/openai-community/gpt2)
- [Hugging Face: Qwen/Qwen3-0.6B (tokeniser used in Lab 1.1)](https://huggingface.co/Qwen/Qwen3-0.6B)

### Adaptation

- [Hu et al.: LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Dettmers et al.: QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Hugging Face PEFT documentation](https://huggingface.co/docs/peft/index)

### Evaluation, services and hosted models

- [Mitchell et al.: Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
- [scikit-learn: StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [Pydantic documentation: validators](https://docs.pydantic.dev/latest/concepts/validators/)
- [OpenRouter: list of models and prices](https://openrouter.ai/api/v1/models)
- [OpenRouter: rate limits](https://openrouter.ai/docs/api-reference/limits)

### National references (read them at their official addresses)

- [Personal Data Protection Law](https://sdaia.gov.sa/en/SDAIA/about/Documents/PersonalDataProtectionLaw.pdf)
- [Implementing Regulation of the Personal Data Protection Law](https://sdaia.gov.sa/en/SDAIA/about/Documents/ImplementingRegulationPersonalDataProtectionLaw.pdf)
- [SDAIA: Data Classification Policy](https://sdaia.gov.sa/en/SDAIA/about/Documents/DataClassificationPolicy.pdf)
- [SDAIA: Principles of AI Ethics](https://sdaia.gov.sa/en/SDAIA/about/Documents/ai-principles.pdf)
- [SDAIA: Generative AI Guidelines for Government](https://sdaia.gov.sa/en/SDAIA/about/Files/GenAIGuidelinesForGovernmentENCompressed.pdf)
- [SDAIA: Generative AI Guidelines for the Public](https://sdaia.gov.sa/en/SDAIA/about/Files/GenerativeAIPublicEN.pdf)
<!-- /academy:sources -->
