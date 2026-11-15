---
course: "Applied Natural Language Processing Engineering"
code: "AE-301"
version: "2027.1.0"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-301/index.html"
---

# Applied Natural Language Processing Engineering

الهندسة التطبيقية لمعالجة اللغات الطبيعية

AI Engineering Track · 3 teaching days + capstone day · 21 hours · Level 300

Route, search and extract customer messages in four varieties of Arabic and English, and choose the model on measured quality, latency and cost

Participants build the language layer of a customer-service system for a fictional regional water utility whose messages arrive in Modern Standard Arabic, Gulf dialect, Arabic written in Latin script and English. They trace text through tokenisers, embeddings and attention, build semantic search, route messages and extract account fields three ways (a classical baseline, a prompted generative model and a fine-tuned encoder), adapt a model with low-rank adapters in free notebook memory, and ship the chosen model as a small service with a model card. Every choice rests on scores by dialect, latency and cost.

**Who it is for:** Practitioners. Engineers who have completed the level 200 set or can show equivalent work: a deployed model or language model application they built and can explain.

**Format:** Intensive: 3 teaching days of four sessions and a capstone day, with guided pre-work and a tracked follow-up task

## The case study

**Sawaqi Water Company.** Sawaqi Water Company is a fictional regional water utility whose customers write in Modern Standard Arabic, Gulf dialect, Arabic in Latin script and English. Participants build Ghayth, its language layer: tokenisers and normalisation, semantic search over 2,000 messages, a routing baseline, grounded replies from the tariff rule book, schema-validated extraction with masking, a fine-tuned encoder and a low-rank adapter, and a small service with a model card. All data is synthetic and generated with a fixed seed; nothing downloads a language model.

## The course

### Day 1: Represent

Slides: https://sdaiaacademy.github.io/ae-301/slides/day1.md

- **Session 1: Tokenisation**
  - Explain how a subword tokeniser splits Arabic, Arabizi and English text
  - Measure tokens per message and price the difference by variety
  - Normalise alef, hamza, ta marbuta, diacritics and digits, and trace a failure to its cause
  - [Lab 1.1: Three tokenisers, four varieties (25 min)](https://sdaiaacademy.github.io/ae-301/labs/D1_1_tokenisers.ipynb)
- **Session 2: Embeddings and semantic search**
  - Explain what an embedding and cosine similarity capture
  - Index a message archive and answer queries with a vector index
  - Score recall at 10 for each language variety and read where it fails
  - [Lab 1.2: Semantic search over 2,000 messages (25 min)](https://sdaiaacademy.github.io/ae-301/labs/D1_2_semantic_search.ipynb)
- **Session 3: Inside the transformer**
  - Describe attention as queries, keys and values
  - Read an attention map for one Arabic sentence
  - Explain when an encoder and when a decoder fits the task
  - [Lab 1.3: Attention in an encoder and a decoder (25 min)](https://sdaiaacademy.github.io/ae-301/labs/D1_3_attention.ipynb)
- **Session 4: Decision models**
  - Build a bag-of-words linear classifier without AI assistance
  - Report macro F1 and F1 by language variety
  - Decide when a classifier should abstain
  - [Lab 1.4: A routing baseline, without AI (25 min)](https://sdaiaacademy.github.io/ae-301/labs/D1_4_baseline.ipynb)

### Day 2: Generate

Slides: https://sdaiaacademy.github.io/ae-301/slides/day2.md

- **Session 1: Generative language models**
  - Explain how temperature and top-p change the next-token choice
  - Measure the consistency of replies across repeated samples
  - Detect details a reply invented
  - [Lab 2.1: Sampling, consistency and invented details (25 min)](https://sdaiaacademy.github.io/ae-301/labs/D2_1_sampling.ipynb)
- **Session 2: Prompting and structured outputs**
  - Write an extraction prompt with a schema and examples
  - Validate replies and repair or log the failures
  - Mask personal fields before text reaches a hosted model
  - [Lab 2.2: Masked, validated extraction (25 min)](https://sdaiaacademy.github.io/ae-301/labs/D2_2_extraction.ipynb)
- **Session 3: Retrieval-augmented generation**
  - Chunk and index a bilingual rule book
  - Write a prompt that cites its sources and refuses when they are silent
  - Evaluate retrieval and generation separately
  - [Lab 2.3: Grounded replies from the rule book (25 min)](https://sdaiaacademy.github.io/ae-301/labs/D2_3_rag.ipynb)
- **Session 4: Model selection and evaluation**
  - Compare models on one labelled set by macro F1 and F1 by variety
  - Measure latency percentiles and cost per thousand messages
  - Defend a choice against a latency budget
  - [Lab 2.4: Two open-weight models against the baseline (25 min)](https://sdaiaacademy.github.io/ae-301/labs/D2_4_model_selection.ipynb)

### Day 3: Adapt

Slides: https://sdaiaacademy.github.io/ae-301/slides/day3.md

- **Session 1: Fine-tuning data**
  - Decide when fine-tuning is worth it
  - Split by customer to prevent leakage
  - Check label balance by department and variety
  - [Lab 3.1: A split without leakage (25 min)](https://sdaiaacademy.github.io/ae-301/labs/D3_1_splits.ipynb)
- **Session 2: Fine-tuning an encoder**
  - Add a classification head to a pretrained encoder
  - Fine-tune it with sensible defaults and watch validation
  - Read a confusion matrix and name the errors
  - [Lab 3.2: Fine-tune the course encoder for routing (25 min)](https://sdaiaacademy.github.io/ae-301/labs/D3_2_finetune_encoder.ipynb)
- **Session 3: Low-rank and quantised adapters**
  - Explain the memory cost of full fine-tuning
  - Write a low-rank adapter layer and count trainable parameters
  - Compare an adapter run with full fine-tuning on the same split
  - [Lab 3.3: A low-rank adapter in a memory budget (25 min)](https://sdaiaacademy.github.io/ae-301/labs/D3_3_lora.ipynb)
- **Session 4: Evaluate and deploy**
  - Put every approach in one comparison table
  - Serve the chosen model behind a small API with a smoke test
  - Write a model card with results by variety and a hand-off rule
  - [Lab 3.4: A small service with a model card (25 min)](https://sdaiaacademy.github.io/ae-301/labs/D3_4_service.ipynb)

### Capstone: Capstone: a routing and extraction service and its defence

Slides: https://sdaiaacademy.github.io/ae-301/slides/day4.md

## Capstone

A routing and extraction service for the utility, or for a comparable text task with synthetic data, submitted as a public repository whose README reports the three-way comparison by dialect, latency and cost, with a model card, then an eight-minute defence and four minutes of questions

**What you submit:** A public GitHub repository whose README reports the three-way comparison (baseline, prompted model, fine-tuned encoder) by dialect, CPU latency and cost per thousand messages, with the model card and timestamped training and evaluation logs committed. Then an eight-minute live defence and four minutes of questions: the participant re-scores the model on a batch the assessor supplies and explains two errors.

**Rubric:**

- Three-way comparison: 25%
- Extraction: 20%
- Adaptation: 20%
- Service and model card: 20%
- Reproducibility: 15%

## Outcomes

- **LO1** Explain how tokenisation, embeddings and attention turn Arabic and English text into model inputs, and where script, diacritics and dialect break that pipeline, given 50 parallel messages and three tokenisers (Awareness) Evidence: a one-page note with token counts per language and two failures traced to their cause
- **LO2** Implement semantic search over a bilingual message archive and measure it for each language variety, given 2,000 synthetic messages and 40 labelled queries (Applied) Evidence: a table of recall at 10 for each of the four varieties
- **LO3** Implement schema-validated extraction of account numbers, districts and complaint types from free text, masking personal fields before text reaches a hosted model, given 200 messages with gold fields (Applied) Evidence: field-level precision and recall, a log of validation failures and a passing masking test
- **LO4** Apply low-rank adapter fine-tuning to an encoder within a fixed memory budget and compare it with full fine-tuning, given a routing set split by customer and a free notebook GPU (Applied) Evidence: training logs, trainable-parameter counts and test scores for both runs
- **LO5** Decide among a classical baseline, a prompted generative model and a fine-tuned encoder for message routing, given eight departments, a held-out test set and a latency budget (Performance) Evidence: a table of macro F1, F1 by dialect, CPU latency and cost per thousand messages, with a defended recommendation
- **LO6** Deliver the chosen model as a tested service with a model card that states its limits by language variety, given a smoke-test set (Performance) Evidence: a running service that passes the smoke test and a model card with results by dialect and a hand-off rule

## Tools

- **Model library and training framework:** Hugging Face Transformers, PyTorch
- **Embeddings and vector index:** Sentence Transformers, FAISS, LanceDB
- **Baselines and metrics:** scikit-learn, seqeval
- **Parameter-efficient fine-tuning:** PEFT, a hand-written low-rank adapter layer
- **Arabic text processing:** CAMeL Tools, PyArabic
- **Language model access:** a locally served open-weight model, a free hosted open-weight model, any commercial API
- **Service, validation and demo:** FastAPI with Pydantic, Gradio, Streamlit

## Prerequisites

- AE-202 Deep Learning and Generative AI Fundamentals
- Trains a small neural network with a hand-written training loop
- Computes precision, recall and F1 from predictions
- Reads and writes UTF-8 Arabic text in Python
- Setup: a laptop CPU with uv, or Google Colab, and a key for a free hosted model (OpenRouter by default). Nothing downloads a language model. Run 00_setup_check before Day 1; see labs/README.md.

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

---

SDAIA Academy · AI Engineering Track · Applied Natural Language Processing Engineering

Course version 2027.1.0 · engine 1.0.0 · built 2026-11-15 · commit 93e55d3
