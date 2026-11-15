---
course: "Applied Natural Language Processing Engineering"
code: "AE-301"
version: "2027.1.0"
scope: "course"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-301/"
---

# Applied Natural Language Processing Engineering

Route, search and extract customer messages in four varieties of Arabic and English, and choose the model on measured quality, latency and cost

Participants build the language layer of a customer-service system for a fictional regional water utility whose messages arrive in Modern Standard Arabic, Gulf dialect, Arabic written in Latin script and English. They trace text through tokenisers, embeddings and attention, build semantic search, route messages and extract account fields three ways (a classical baseline, a prompted generative model and a fine-tuned encoder), adapt a model with low-rank adapters in free notebook memory, and ship the chosen model as a small service with a model card. Every choice rests on scores by dialect, latency and cost.

- **Audience:** Practitioners. Engineers who have completed the level 200 set or can show equivalent work: a deployed model or language model application they built and can explain.
- **Prerequisites:** AE-202 Deep Learning and Generative AI Fundamentals; Trains a small neural network with a hand-written training loop; Computes precision, recall and F1 from predictions; Reads and writes UTF-8 Arabic text in Python; Setup: a laptop CPU with uv, or Google Colab, and a key for a free hosted model (OpenRouter by default). Nothing downloads a language model. Run 00_setup_check before Day 1; see labs/README.md.
- **Case study:** Sawaqi Water Company: Sawaqi Water Company is a fictional regional water utility whose customers write in Modern Standard Arabic, Gulf dialect, Arabic in Latin script and English. Participants build Ghayth, its language layer: tokenisers and normalisation, semantic search over 2,000 messages, a routing baseline, grounded replies from the tariff rule book, schema-validated extraction with masking, a fine-tuned encoder and a low-rank adapter, and a small service with a model card. All data is synthetic and generated with a fixed seed; nothing downloads a language model.

## Outcomes

- LO1 (Awareness): Explain how tokenisation, embeddings and attention turn Arabic and English text into model inputs, and where script, diacritics and dialect break that pipeline, given 50 parallel messages and three tokenisers
- LO2 (Applied): Implement semantic search over a bilingual message archive and measure it for each language variety, given 2,000 synthetic messages and 40 labelled queries
- LO3 (Applied): Implement schema-validated extraction of account numbers, districts and complaint types from free text, masking personal fields before text reaches a hosted model, given 200 messages with gold fields
- LO4 (Applied): Apply low-rank adapter fine-tuning to an encoder within a fixed memory budget and compare it with full fine-tuning, given a routing set split by customer and a free notebook GPU
- LO5 (Performance): Decide among a classical baseline, a prompted generative model and a fine-tuned encoder for message routing, given eight departments, a held-out test set and a latency budget
- LO6 (Performance): Deliver the chosen model as a tested service with a model card that states its limits by language variety, given a smoke-test set

## Day 1: Represent

#### [day1#title] Represent

*SDAIA Academy · Applied Natural Language Processing Engineering*

Day 1

AI Engineering Track · 3 teaching days + capstone day · 4 sessions · 4 labs · AE-301

#### [day1#agenda] Represent

*Today*

- **Session 1 · Tokenisation**: Text becomes integers: subword tokenisers, their cost per language, and the Arabic letters that break them
- **Session 2 · Embeddings and semantic search**: Messages as vectors: index once, query by meaning, and measure recall for each variety
- **Session 3 · Inside the transformer**: Attention, layers and the two families: an encoder reads both ways, a decoder only looks back
- **Session 4 · Decision models**: Most production language work is classification: start with a baseline and report it by subgroup

#### [day1#sawaqi-and-ghayth] Sawaqi Water Company gets 2,000 messages you will learn to read

*The case for four days*

**MSA:** انقطعت المياه عن منزلنا في حي النسيم منذ الصباح

**Gulf:** الماي مقطوع عندنا في النسيم من الصبح

**Arabizi:** el mai maqtoo3 3indna fi el naseem min el sob7

**English:** We have had no water at home in Al-Naseem since the morning

- **Ghayth, the language layer.** Route each message to one of eight departments, extract the account number, district and complaint type, find similar messages, draft replies from the tariff rule book.
- **Four varieties, one customer.** Every result in this course is reported for each variety. An average hides who is failed.

#### [day1#three-ways-one-table] Three ways to route a message, one table to choose with

*Where the course goes*

| Approach | Built on | Macro F1 | F1 by variety | CPU latency | Cost per 1,000 |
| --- | --- | --- | --- | --- | --- |
| Classical baseline | Day 1 | you fill these cells with your own runs, and defend a choice on Day 4 |  |  |  |
| Prompted generative model | Day 2 |  |  |  |  |
| Fine-tuned encoder | Day 3 |  |  |  |  |

> **The habit to take home.** No model is chosen on a single number, and none without a baseline next to it.

### Session 1: Tokenisation

#### [day1#s1] Tokenisation

*Session 1*

- Explain how a subword tokeniser splits Arabic, Arabizi and English text
- Measure tokens per message and price the difference by variety
- Normalise alef, hamza, ta marbuta, diacritics and digits, and trace a failure to its cause

#### [day1#models-read-integers] A model never sees text: it sees a list of integers

*Tokenisation*

1. **Text** The meter is broken
2. **Pieces** The · meter · is · broken
3. **Ids** 785 · 16430 · 318 · 5445
4. **Vectors** one row of a table per id

- The **tokeniser** is fixed before training and shipped with the model: change it and the model reads noise.
- Everything a provider counts and charges is tokens, not words or characters.

#### [day1#subwords-between] Subwords sit between characters and words

*The unit*

| Unit | Vocabulary | A new word | Length of a message |
| --- | --- | --- | --- |
| Characters or bytes | tiny | always representable | very long |
| Whole words | millions, never complete | unknown | short |
| Subwords | 30,000 to 250,000 | built from pieces | in between |

`token` `is` `ation` `ال` `فات` `ورة`

A frequent word is one piece. A rare word is several. A word from a language the tokeniser rarely saw is many.

#### [day1#bpe-merges] Byte-pair encoding merges the most frequent pair, again and again

*How a vocabulary is learnt*

1. Start from single characters or bytes.
2. Count every adjacent pair in the training text.
3. Merge the most frequent pair into one new piece.
4. Repeat until the vocabulary has the size you asked for.

- **What the merges remember.** The training text. English web text gives English pieces; Arabic text gives Arabic pieces.
- **WordPiece.** The same idea with another score; it marks a continuing piece with `##` and has an `[UNK]` piece.

#### [day1#tokens-per-variety] The same complaint costs a different number of tokens in each variety

*Measured in Lab 1.1 · 50 parallel messages*

| Tokeniser | MSA | Gulf | Arabizi | English |
| --- | --- | --- | --- | --- |
| GPT-2 (English web text) | 40.1 | 32.7 | 18.8 | 11.8 |
| Qwen3 (multilingual) | 15.0 | 14.6 | 19.4 | 11.9 |
| CAMeLBERT-mix (Arabic) | 9.7 | 8.7 | 25.5 | 23.6 |

Mean tokens per message. Same facts in every column.

> **No row is good at Arabizi.** Nobody trained a tokeniser on it, so every tokeniser spells it out.

#### [day1#gpt2-falls-back-to-bytes] A tokeniser that never saw Arabic falls back to bytes

*Why 40 tokens*

- **5.3** GPT-2 tokens per MSA word
- **2.0** Qwen3 tokens per MSA word
- **1.3** CAMeLBERT tokens per MSA word

- An Arabic letter is two bytes in UTF-8. With no merged pieces, a five-letter word is up to ten tokens.
- Nothing breaks and nothing is unknown: the message is only three times longer, slower and more expensive.

#### [day1#price-per-thousand] Price the difference, then look at what you are paying for

*Tokens are money*

| Variety | Message only | With a 180-token prompt |
| --- | --- | --- |
| MSA | 0.0030 | 0.0390 |
| Gulf | 0.0029 | 0.0389 |
| Arabizi | 0.0039 | 0.0399 |
| English | 0.0024 | 0.0384 |

- **Message only.** Arabizi costs 1.62 times English.
- **With the instructions.** The prompt is over 90% of the bill in every variety.

USD per 1,000 messages at 0.20 USD per million input tokens, Qwen3 token counts.

#### [day1#one-word-many-spellings] One word, many spellings: normalise before you match or count

*Arabic orthography*

| What varies | As written | Normalised to |
| --- | --- | --- |
| Diacritics and shadda | العَدّاد | العداد |
| Tatweel, the stretching mark | العـداد | العداد |
| Alef with hamza | أرجو · إغلاق | ارجو · اغلاق |
| Ta marbuta written as ha | الفاتورة · الفاتوره | الفاتوره |
| Alef maqsura for ya | فى | في |
| Arabic-Indic digits | ٣٠٦١٠٧٤٦١٠ | 3061074610 |

> **Normalise for matching, keep the original for display.** The customer wrote it their way.

#### [day1#two-failures-traced] Two failures, each traced to its cause

*Measured in Lab 1.1*

- **One diacritic, one unknown word.** العَدّاد becomes `[UNK]` for the Arabic tokeniser and 9 pieces instead of 5 for the multilingual one. **Cause.** The training text had no diacritics, so no piece contains one.
- **An account number at twice the price.** A sentence with ٣٠٦١٠٧٤٦١٠ costs 25 tokens; with Western digits, 15. **Cause.** Each Arabic-Indic digit is two bytes with no merged piece.

- **1,052 → 0** unknown pieces in 1,340 Arabic-script messages, before and after normalising
- **2,968 → 1,760** distinct word forms, before and after

#### [day1#arabizi-digits-for-letters] Arabizi has no spelling, so every writer invents one

*Dialect and Latin script*

| Digit | Letter | Example |
| --- | --- | --- |
| 3 | ع | 3adad (meter) |
| 7 | ح | 7sab (account) |
| 5 | خ | 5azan (tank) |
| 6 · 9 | ط · ص | 6ag6 (pressure) · 9a3eef (weak) |

- The same word arrives as **3adad**, **aadad**, **adad**: the archive has all three.
- Dialect changes the words too: الماء is الماي or المويه in Gulf messages.
- No tokeniser, and few models, were trained on it. Expect it to be the worst column of every table this week.

#### [day1#check-tokenisers] Which change cuts the token bill most?

*Checkpoint*

**Question:** Sawaqi routes 60,000 messages a month with a hosted model. Each call sends 180 tokens of instructions and a message of about 15 tokens. Which single change cuts the bill most?

- A. Translating Arabizi messages to English first
- B. Removing diacritics and tatweel from every message
- C. Shortening the instructions from 180 to 90 tokens
- D. Switching the hosted model to an Arabic tokeniser

#### [day1#lab-1-1] Three tokenisers, four varieties

Lab 1.1 · 25 min

Count tokens for 50 parallel Arabic, dialect and English messages with three tokenisers and price the difference, then normalise and trace two failures to their cause.

[D1_1_tokenisers.ipynb](https://sdaiaacademy.github.io/ae-301/labs/D1_1_tokenisers.ipynb)

- [ ] Load three tokenisers and split one message (4 min)
- [ ] Tokens per message: 50 messages, four varieties (7 min)
- [ ] Price the difference per thousand messages (5 min)
- [ ] Normalise, then trace two failures to their cause (9 min)
- [ ] ✱ Stretch: Train a small tokeniser on the Sawaqi messages and count again

### Session 2: Embeddings and semantic search

#### [day1#s2] Embeddings and semantic search

*Session 2*

- Explain what an embedding and cosine similarity capture
- Index a message archive and answer queries with a vector index
- Score recall at 10 for each language variety and read where it fails

#### [day1#meaning-as-a-point] An embedding puts a text at a point; close points mean close meanings

*Embeddings*

- A token id selects a row of a table: its **embedding**.
- A model mixes the rows of a message into one vector for the whole message.
- Trained so that texts used alike land close together.
- A **multilingual** model lands a message and its translation close together.

- **Close.** الماي مقطوع · no water at home · el mai maqtoo3
- **Far.** no water at home · I was charged twice

#### [day1#cosine-is-a-dot-product] With unit vectors, similarity is one dot product

*Similarity*

```python
# file: Lab 1.2
# E: (2000, d), each row of length 1
E = G.embed(messages)
q = G.embed([query])[0]   # (d,)
# cosine similarity, 2000 at once
scores = E @ q
top = np.argsort(-scores)[:10]
```

- **Cosine similarity** compares directions and ignores length: 1 the same, 0 unrelated.
- Normalise every vector to length 1 once; then cosine is the dot product.
- One matrix multiplication scores a query against the whole archive.

#### [day1#choosing-an-embedding-model] Choose an embedding model on your own queries, in your own varieties

*Which model*

| Question | Why it matters at Sawaqi |
| --- | --- |
| Which languages was it trained on? | An English-only model sees Arabic as noise |
| Are the weights open, and under which licence? | You may need to run it yourself later |
| How long a text does it take, how many dimensions? | Messages are short; memory is rows × dimensions |
| Where does it run, and what leaves the building? | Message text carries personal data |
| What does it score on **your** 40 queries? | A public leaderboard has no Gulf Arabizi |

#### [day1#index-once-query-often] Index once, offline; answer every query online

*The shape of every search system*

1. **Offline** normalise and embed 2,000 messages
2. **Index** store the vectors with their message ids
3. **Online** embed the query with the same model
4. **Search** top 10 by similarity

- **Flat index.** Compare with every row. Exact. Fine to about a million rows.
- **Approximate index.** Visit a fraction of the rows (FAISS, LanceDB). Faster, and it can miss: measure its recall too.

#### [day1#recall-at-ten] Recall at 10: of the relevant messages a top-10 list could hold, how many it holds

*The metric*

- query: el majari 6af7a fi el rabwa
- relevant: 8 messages: sewage overflow in Al-Rabwa, in any variety
- top 10: holds 2 of them
- recall at 10: 2 / min(10, 8) = 0.25

- Divide by `min(10, relevant)`: a list of ten cannot hold fifteen.
- Average over the queries of one variety, then report the four numbers side by side.
- Relevance here is a rule (same complaint type, same district). In production it is a person's judgement, and it costs time.

#### [day1#ngram-baseline-by-variety] A character n-gram baseline finds its own script and little else

*Measured in Lab 1.2 · 20 queries*

| Item | Value |
| --- | --- |
| English | 0.40 |
| MSA | 0.33 |
| Gulf | 0.29 |
| Arabizi | 0.02 |

- Recall at 10 of TF-IDF over character 2- to 4-grams, no model.
- The relevant messages are in all four varieties; shared letters find only the query's own.
- Arabizi: every writer spells differently, so even its own script fails.
- **Your embedding model's row goes next to this one.** That table is the evidence.

#### [day1#where-meaning-search-fails] Semantic search fails where there is no meaning to embed

*Know the tool's edge*

- **Identifiers.** An account number, a ticket, a meter serial. Use an exact lookup.
- **Negation and numbers.** "No leak" and "a leak" are close. So are 14 days and 41 days.
- **Rare varieties.** What the model rarely saw lands in the wrong place. Measure it per variety.

> **Hybrid.** Run an exact or keyword search beside the vector search and merge the two rankings. The lab's stretch does it in eight lines.

#### [day1#check-semantic-search] An agent searches for an account number and gets nonsense

*Checkpoint*

**Question:** Hessa's agents type an account number into Ghayth's semantic search and get ten unrelated messages. The same search works well for "brown water in Al-Qalaa". What do you change?

- A. Add an exact lookup on the account number beside the vector search
- B. Move to a larger embedding model
- C. Replace the approximate index by a flat one
- D. Embed each message together with its customer's name

#### [day1#lab-1-2] Semantic search over 2,000 messages

Lab 1.2 · 25 min

Index 2,000 messages with an open multilingual embedding model and score 20 queries by variety, against a character n-gram baseline.

[D1_2_semantic_search.ipynb](https://sdaiaacademy.github.io/ae-301/labs/D1_2_semantic_search.ipynb)

- [ ] Normalise and build a character n-gram baseline (6 min)
- [ ] Embed 2,000 messages with a hosted open model (6 min)
- [ ] A flat inner-product index and one query (5 min)
- [ ] Recall at 10 for 20 queries, by variety (8 min)
- [ ] ✱ Stretch: Score all 40 queries and add a hybrid of both rankings

### Session 3: Inside the transformer

#### [day1#s3] Inside the transformer

*Session 3*

- Describe attention as queries, keys and values
- Read an attention map for one Arabic sentence
- Explain when an encoder and when a decoder fits the task

#### [day1#attention-ask-offer-take] Every token asks, every token offers, and each takes a weighted mix

*Attention*

- **Query.** What this token is looking for.
- **Key.** What each token offers to be found by.
- **Value.** What a token hands over when it is chosen.

```python
# file: one head
weights = softmax(Q @ K.T / sqrt(d_k))   # (tokens, tokens): each row sums to 1
output  = weights @ V                     # each token: a weighted average of the values
```

#### [day1#attention-by-hand] Attention is eight lines, and the lab checks yours against the model's

*Lab 1.3, step 2*

```python
# file: Lab 1.3
def attention_weights(q, k, causal=False):
    scores = q @ k.T / math.sqrt(q.shape[-1])
    if causal:      # a decoder may not look ahead
        seen = torch.ones_like(scores).tril().bool()
        scores = scores.masked_fill(~seen, -math.inf)
    return scores.softmax(-1)
```

- Dividing by the square root of the head's width keeps the softmax from saturating.
- The **causal mask** is the only difference between an encoder's attention and a decoder's.
- The check: your weights equal the model's own for layer 0, head 0, to five decimals.

#### [day1#heads-layers-blocks] A transformer is that operation, several heads wide and several layers deep

*The whole model*

1. **Token and position embeddings** one vector per token
2. **Attention, several heads** each head looks for something else
3. **Feed-forward layer** each token on its own
4. **Repeat for every layer** then a head for the task

| The course encoder | Value |
| --- | --- |
| Layers × heads | 2 × 4 |
| Width | 128 |
| Vocabulary | 2,000 |
| Parameters | 529,408 |
| Of which token embeddings | 256,000 |

#### [day1#encoder-sees-both-ways] An encoder reads both ways; a decoder only looks back

*One mask apart*

- **Encoder.** Every token attends to every token. In the lab's sentence, 36% of the first layer's attention goes to later tokens. **Trained to** fill in a hidden token from both sides.
- **Decoder.** A token attends to itself and earlier tokens. Attention to later tokens: exactly 0. **Trained to** predict the next token from the left side.

> **An attention map shows where a head looked.** It is not an explanation of the model's decision.

#### [day1#fill-gap-or-continue] Fill the gap, or predict what comes next

*Measured in Lab 1.3*

- encoder: the water [MASK] is very weak on the second floor
- fills: "pressure" first: it has read the rest of the sentence

- decoder: the water
- continues: several openings stay likely: smells, coming, pressure, was

- A decision about a whole message wants the whole message in view: an **encoder** with a small head.
- Writing text token by token is what a **decoder** is for; it can also be asked to decide, at a price.

#### [day1#which-family-for-which-job] Match the family to the job before you match the model

*Choosing a family*

| Job at Sawaqi | Family | Why |
| --- | --- | --- |
| Route a message to one of eight departments | Encoder + classification head | One pass, eight scores, milliseconds on a processor |
| Tag each word (a district, an account number) | Encoder + token-classification head | One score per token |
| Find similar messages | Encoder trained for embeddings | One vector per message |
| Draft a reply, extract into JSON without training data | Decoder, prompted | Writes text; needs no labels; costs per token |

#### [day1#check-which-family] Two million messages a day, eight departments

*Checkpoint*

**Question:** A national utility must label 2 million messages a day with one of eight departments, on its own servers without graphics processors, and has 50,000 labelled messages. Which design fits?

- A. A large prompted decoder that writes the department name
- B. An encoder-decoder that translates the message into a label
- C. An embedding model and a search over all labelled messages
- D. A small encoder with a classification head, fine-tuned on the labels

#### [day1#lab-1-3] Attention in an encoder and a decoder

Lab 1.3 · 25 min

Inspect attention for one Arabic sentence in the course encoder and decoder, compare what each token can see, and compare filling a gap with predicting the next token.

[D1_3_attention.ipynb](https://sdaiaacademy.github.io/ae-301/labs/D1_3_attention.ipynb)

- [ ] Load the course encoder and decoder (4 min)
- [ ] Attention by hand for one head (7 min)
- [ ] Read the maps: who attends to whom (7 min)
- [ ] Fill the gap, or predict the next token (7 min)
- [ ] ✱ Stretch: Change one letter of the sentence and compare the maps

### Session 4: Decision models

#### [day1#s4] Decision models

*Session 4*

- Build a bag-of-words linear classifier without AI assistance
- Report macro F1 and F1 by language variety
- Decide when a classifier should abstain

#### [day1#most-nlp-is-a-decision] Most language work in production is a decision, not a paragraph

*Decision models*

- **Classification.** One label for a message: a department, urgent or not, a language variety.
- **Token classification.** One label per word: this is a district, this is an account number.
- **Ranking.** An order over candidates: which earlier messages are most alike.

- A decision has a right answer, so it can be scored on thousands of examples.
- It runs on every message, so its latency and cost are multiplied by the volume.

#### [day1#baseline-first] Build the cheapest thing that could work, and make everything else beat it

*Baselines first*

- It takes an hour and sets the floor.
- It shows whether the labels and the split are sound before a larger model hides the problem.
- It is the fallback when the larger model is down, slow or withdrawn.
- Sometimes it wins. Then you are done.

- **Without a baseline.** "The model scores 0.84" means nothing. Is that 20 points above word counts, or 1 point below?

#### [day1#words-and-character-ngrams] Count words and character n-grams; weigh the rare ones up

*The baseline's eyes*

```python
# file: Lab 1.4
make_union(
    # words and pairs of words
    TfidfVectorizer(preprocessor=G.normalise,
                    ngram_range=(1, 2)),
    # pieces of words, 2 to 4 characters
    TfidfVectorizer(preprocessor=G.normalise,
                    analyzer="char_wb",
                    ngram_range=(2, 4)))
# then LogisticRegression(C=10, max_iter=1000)
```

- **TF-IDF**: a word counts more when few messages contain it.
- **Character n-grams** survive a typo, a glued prefix (بالفاتوره) and an Arabizi spelling that varies.
- A **linear model** gives each feature one weight per department: you can read why it decided.
- 400 messages give 13,005 features.

#### [day1#macro-f1-by-hand] Macro F1 gives every department one vote, however small

*The metric, by hand*

| Department | Precision | Recall | F1 |
| --- | --- | --- | --- |
| billing (large) | 0.90 | 0.90 | 0.90 |
| supply (large) | 0.80 | 1.00 | 0.89 |
| connections (small) | 1.00 | 0.20 | 0.33 |
| Macro F1 |  |  | 0.71 |

- **Precision**: of the messages sent to a department, how many belonged there.
- **Recall**: of the messages that belonged there, how many arrived.
- **F1** = 2PR / (P + R). **Macro**: the plain mean over departments.
- Accuracy would hide the small department; macro F1 puts it on the slide.

#### [day1#baseline-by-variety] The baseline scores 0.85, and half of that for Arabizi

*Measured in Lab 1.4 · 403 test messages*

| Item | Value |
| --- | --- |
| All | 0.85 |
| MSA | 0.94 |
| Gulf | 0.86 |
| English | 0.81 |
| Arabizi | 0.42 |

**4 ms** median per message on one laptop processor; 5 ms at the 95th percentile

- **Report every row.** "0.85" is true and hides one customer in eight.

#### [day1#no-examples-no-skill] A model cannot learn a pair it has no examples of

*Why Arabizi fails*

| Department | Arabizi messages in training | Arabizi test messages right |
| --- | --- | --- |
| supply | 11 | 8 of 16 |
| billing | 7 | 9 of 10 |
| meters | 4 | 4 of 7 |
| quality | 2 | 0 of 3 |
| wastewater | 0 | 0 of 2 |

- Accuracy on Arabizi is 0.57; macro F1 is 0.42, because the empty departments count in full.
- The fix is not a cleverer model first. It is labels where there are none, or a person for those messages.

#### [day1#abstain-and-hand-off] Below a confidence threshold, hand the message to a person

*A router may say "I do not know"*

| Threshold | Handed off | Accuracy on the rest |
| --- | --- | --- |
| 0.3 | 40 of 403 | 0.94 |
| 0.4 | 79 | 0.97 |
| 0.5 | 128 | 0.98 |
| 0.6 | 168 | 1.00 |

- Every point on this table is a staffing decision: people cost more than mistakes, or the reverse.
- At 0.5, 31 of the 51 Arabizi messages go to a person.
- The threshold belongs in the model card, with its hand-off rate. Day 3 writes it there.

#### [day1#check-macro-f1] One number went up. Did the router get better?

*Checkpoint*

**Question:** A new version raises test macro F1 from 0.85 to 0.87: MSA 0.94 to 0.97, Gulf 0.86 to 0.90, English 0.81 to 0.84, Arabizi 0.42 to 0.31. What do you report?

- A. An improvement of two points, ready to ship
- B. A gain for three varieties, a loss for Arabizi, and a hand-off rule for Arabizi
- C. No real change, since the Arabizi test set is too small to matter
- D. A regression that must be rolled back in full

#### [day1#lab-1-4] A routing baseline, without AI

Lab 1.4 · 25 min

Train a bag-of-words linear baseline for routing without AI assistance and report macro F1 by dialect, with predictions and CPU latency saved for Day 2.

[D1_4_baseline.ipynb](https://sdaiaacademy.github.io/ae-301/labs/D1_4_baseline.ipynb)

- [ ] Features: word and character n-grams (6 min)
- [ ] Train a linear model on the customer split (5 min)
- [ ] Macro F1 overall and by dialect (7 min)
- [ ] Save predictions and CPU latency (7 min)
- [ ] ✱ Stretch: Abstain below a confidence threshold and count the hand-offs

#### [day1#day-1-recap] Text became numbers, and you have a number to beat

*Day 1 recap*

- **Tokens** are the unit of cost; the same message costs up to three times more in a variety the tokeniser never saw.
- **Normalise** before matching: 1,052 unknown pieces became 0.
- **Embeddings** search by meaning; identifiers need an exact lookup.

- **Attention**: an encoder reads both ways, a decoder looks back.
- **Baseline**: macro F1 0.85, Arabizi 0.42, 4 ms a message.
- **Report by variety**, always.

#### [day1#tomorrow-generate] Generate: replies, extraction and a prompted router, each with a measure

*Tomorrow · Day 2*

Sample from a decoder and count what it invents, mask personal fields and extract into a validated schema, ground replies in the tariff rule book, and put two open-weight models in one table with today's baseline.

## Day 2: Generate

#### [day2#title] Generate

*SDAIA Academy · Applied Natural Language Processing Engineering*

Day 2

AI Engineering Track · 3 teaching days + capstone day · 4 sessions · 4 labs · AE-301

#### [day2#agenda] Generate

*Today*

- **Session 1 · Generative language models**: A decoder in a loop: sampling settings, consistency and invented details
- **Session 2 · Prompting and structured outputs**: A prompt is a specification; a schema is the contract; personal fields are masked before they leave
- **Session 3 · Retrieval-augmented generation**: Ground replies in the tariff rule book, cite the rule, and refuse when the rules are silent
- **Session 4 · Model selection and evaluation**: Choose on quality by variety, latency and cost per thousand messages, not on a leaderboard

#### [day2#review-day-1] Before we generate anything

*Review · Day 1*

**Question:** A message contains العَدّاد with diacritics. The Arabic tokeniser returns `[UNK]` and the router sends the message to the wrong department. Where is the fix?

- A. Retrain the router with more epochs
- B. Ask for a tokeniser with a larger vocabulary
- C. Leave it, since diacritics are rare in customer messages
- D. Normalise the text before the tokeniser sees it

### Session 1: Generative language models

#### [day2#s1] Generative language models

*Session 1*

- Explain how temperature and top-p change the next-token choice
- Measure the consistency of replies across repeated samples
- Detect details a reply invented

#### [day2#decoder-in-a-loop] A generative model is yesterday's decoder in a loop

*Generative language models*

1. **Context** the tokens so far
2. **Decoder** one score per vocabulary entry
3. **Sampling** scores become one chosen token
4. **Append** and go round again

- The model never writes a sentence. It scores the next token, thousands of times.
- Everything it "knows" at a step is in the context: your instructions, the message, and what it has written so far.
- It stops at an end token or at your `max_tokens`.

#### [day2#instructions-are-tokens-too] A chat is one long text with markers the model was trained to follow

*Base and instruction-tuned*

- system: You write short replies for the contact centre of Sawaqi. Use only the facts given.
- user: Customer message: … FACTS: {"ticket": "SW-48211", "crew_eta_hours": 4}
- assistant: the model continues from here

- A **base** model continues any text. An **instruction-tuned** model was trained further on conversations in one fixed format.
- The provider's **chat template** turns your list of messages into that format. Use the API's roles; do not write the markers yourself.
- The system message is not a law. It is text the model was trained to weigh heavily.

#### [day2#temperature-sharpens-or-flattens] Temperature sharpens or flattens the same scores

*Measured in Lab 2.1 on the course decoder*

| Temperature | Probability of the top token | Entropy (bits) | Reads as |
| --- | --- | --- | --- |
| 0.2 | 0.996 | 0.04 | almost always the same token |
| 0.7 | 0.631 | 2.42 | the favourite, with variety |
| 1.0 | 0.335 | 4.97 | the model's own distribution |
| 1.5 | 0.108 | 7.63 | unlikely tokens get drawn |

Next token after الماي ("the water"). The scores are divided by the temperature before the softmax. At 0, providers take the top token.

#### [day2#top-p-cuts-the-tail] Top-p removes the long tail before the draw

*Nucleus sampling*

```python
# file: Lab 2.1
probs = (logits / temperature).softmax(-1)
order = probs.argsort(descending=True)
before = probs[order].cumsum(0) - probs[order]
# the smallest set that reaches top_p
keep = order[before < top_p]
out = torch.zeros_like(probs)
out[keep] = probs[keep]
out = out / out.sum()
```

- Thousands of unlikely tokens together hold real probability. One of them, drawn, is the odd word in a reply.
- At temperature 1.0, `top_p=0.9` leaves 73 tokens of 2,000; `top_p=0.5` leaves 3.
- Use one knob at a time, and write down both.

#### [day2#consistency-is-measurable] Consistency is a number: generate twice, compare

*Lab 2.1, steps 2 and 3*

| Setting | Temperature | Top-p | Exact agreement · word overlap |
| --- | --- | --- | --- |
| focused | 0.0 | 1.0 | measured by your run: ten tasks, two samples each |
| balanced | 0.7 | 0.9 |  |
| loose | 1.3 | 1.0 |  |

- **Exact agreement**: the share of sample pairs with identical text.
- **Word overlap**: shared words over all words of the pair.
- A reply a customer could receive in two versions is a complaint waiting to happen. Decide how much variation the task tolerates, then set the knobs.

#### [day2#invented-details] An invented detail is anything stated that the inputs do not hold

*Lab 2.1, step 4*

**Facts:** ticket SW-48211, crew within 4 hours

**Reply (good example):** Your request is logged as SW-48211. A crew will reach you within 4 hours.

**Reply (weak example):** Your request is logged as SW-48211. You will receive a credit of 50 SAR under rule R-11.

- The model is completing a plausible reply. A credit is plausible.
- A regular expression catches the costly kind: **numbers and identifiers** that are in neither the message nor the facts.
- It does not catch an invented promise in words. That needs a person or a second model, and a sample to read.

#### [day2#settings-follow-the-task] Set the knobs by what a wrong output costs

*Choosing settings*

| Task | Setting | Then check with code |
| --- | --- | --- |
| Route to a department | temperature 0, a dozen output tokens | the label is one of eight |
| Extract fields | temperature 0, JSON mode | schema and validators |
| Answer from the rule book | temperature 0 | citation present, refusal exact |
| Draft a reply for an agent to send | moderate temperature, top-p below 1 | no number or identifier outside the facts |

> **Temperature 0 makes a model repeat itself, not tell the truth.** A repeated invented detail is still invented.

#### [day2#check-sampling] The same message, a different department each run

*Checkpoint*

**Question:** Ghayth routes with a prompted model. The same message is sent to billing on one run and to meters on the next. What do you change first?

- A. Add ten examples to the prompt
- B. Set the temperature to 0 and limit the output to a label
- C. Move to a larger model
- D. Call the model three times and take the majority

#### [day2#lab-2-1] Sampling, consistency and invented details

Lab 2.1 · 25 min

Vary sampling settings on ten replies and record consistency and invented details, starting from the course decoder's own next-token distribution.

[D2_1_sampling.ipynb](https://sdaiaacademy.github.io/ae-301/labs/D2_1_sampling.ipynb)

- [ ] Temperature on one next-token distribution (5 min)
- [ ] Ten replies at three settings (8 min)
- [ ] Consistency across samples (6 min)
- [ ] An invented-detail checker (6 min)
- [ ] ✱ Stretch: Fix a seed and see what it does and does not make repeatable

### Session 2: Prompting and structured outputs

#### [day2#s2] Prompting and structured outputs

*Session 2*

- Write an extraction prompt with a schema and examples
- Validate replies and repair or log the failures
- Mask personal fields before text reaches a hosted model

#### [day2#prompt-is-a-specification] A prompt is a specification: say what, from what, and in which shape

*Prompting*

Prompt for Extraction prompt, Lab 2.2:

```text
You extract fields from one customer message of a water utility.
The message may be in Modern Standard Arabic, Gulf dialect, Arabic in Latin script or English. Personal values are masked as [ACCOUNT_n], [PHONE_n], [NAME_n].
Return JSON only, with exactly these keys: account_number, district (one of eight, or null), complaint_type (one of sixteen).
Never invent a value.
```

Task · Input, and what is unusual about it · Output contract · The rule that code will check

#### [day2#mask-before-it-leaves] Mask personal fields before the text leaves, and put them back after

*Personal data*

- message: انا عبدالله الشمري، ارغب في تقسيط مبلغ 425 ريال، حسابي رقم ٣٠٨٢٨٢٨٥١٣
- masked: انا [NAME_1] [NAME_2]، ارغب في تقسيط مبلغ 425 ريال، حسابي رقم [ACCOUNT_1]
- model: `{"account_number": "[ACCOUNT_1]", "district": null, "complaint_type": "installment_request"}`
- unmasked: `account_number = "3082828513"`, inside Sawaqi

- The model needs to know **where** the account number is, not what it is.
- Sawaqi's rule R-10 classifies customer records as Restricted and requires this step.

#### [day2#masking-has-a-test] A masking step without a test is a hope

*Measured in Lab 2.2*

- **200** messages masked
- **0** with a 10-digit run or a known name left
- **200** restored exactly by unmasking

- Arabic-Indic digits are turned into Western digits first: a pattern for `30` and eight digits must find ٣٠ too.
- The test proves the patterns on these messages. It cannot see a name outside the list or a phone number written with spaces: say so in the card.

#### [day2#schema-is-the-contract] The schema is the contract; validators check values, not only shapes

*Structured outputs*

```python
# file: Lab 2.2
class Extraction(BaseModel, extra="forbid"):
    account_number: Optional[str]
    district: Optional[District]        # one of eight
    complaint_type: ComplaintType       # one of sixteen

    @field_validator("account_number")
    @classmethod
    def known_account(cls, value, info):
        mapping = info.context["mapping"]
        if value is not None and value not in mapping:
            raise ValueError("must be an [ACCOUNT_n] "
                             "placeholder from the message")
        return value
```

- **Types** reject a wrong shape: a missing key, a district outside the list.
- The **validator** rejects a wrong value: an account number the model never saw, a phone placeholder in the account field.
- The error message is written for the model: it is sent back in the repair call.

#### [day2#validate-repair-log] Validate, repair once, then log and hand off

*The loop around the model*

1. **Mask** placeholders in
2. **Ask** temperature 0, JSON mode
3. **Validate** schema and validators
4. **Repair once** send the error back
5. **Log** still failing: to a person

- **The log is evidence.** Message id, attempt, error. It shows what the validators stopped, and which errors repair does not fix.
- **Never guess.** A message that fails twice gets no fields. A wrong account number is worse than none.

#### [day2#valid-json-is-not-right] Valid JSON every time is a format guarantee, not a correctness guarantee

*Constrained output*

- **JSON mode** asks the model for JSON; **constrained decoding** removes every token that would break your schema.
- Both end parse errors. Neither knows which district the customer lives in.
- So measure values against gold fields: **precision** and **recall** for each field.

| For one field | Counts as |
| --- | --- |
| Right value returned | correct |
| Value returned, gold has none | costs precision |
| Null returned, gold has a value | costs recall |
| Wrong value returned | costs both |

#### [day2#regex-baseline-fields] The baseline for extraction is three regular expressions, and it has two bugs

*Measured in Lab 2.2 · 200 messages with gold fields*

| Field | Precision | Recall |
| --- | --- | --- |
| Account number | 0.69 | 0.76 |
| District | 0.97 | 0.97 |
| Complaint type (keywords) | 0.58 | 0.58 |

- A number in Arabic-Indic digits is returned as written and does not equal the gold value.
- A message with a phone number and no account number returns the phone number.
- The district is easy; the complaint type is where a model can earn its cost.
- **Your model's rows go beside these.**

#### [day2#check-structured-outputs] The provider now guarantees schema-valid JSON

*Checkpoint*

**Question:** You switch on constrained decoding. Every reply now parses and matches the schema. Which check can you still not drop?

- A. That every reply has the three keys
- B. That the district is one of the eight in the list
- C. That the account placeholder is one that occurs in this message
- D. That the reply is JSON and nothing else

#### [day2#lab-2-2] Masked, validated extraction

Lab 2.2 · 25 min

Extract account number, district and complaint type into a validated schema, mask personal fields before the hosted model sees them, and log failures.

[D2_2_extraction.ipynb](https://sdaiaacademy.github.io/ae-301/labs/D2_2_extraction.ipynb)

- [ ] Mask personal fields, with a test (6 min)
- [ ] A schema whose validators catch wrong values (5 min)
- [ ] Extract with a hosted model, repair once, log failures (8 min)
- [ ] Field precision and recall against a regex baseline (6 min)
- [ ] ✱ Stretch: Add two examples per variety to the prompt and score again

### Session 3: Retrieval-augmented generation

#### [day2#s3] Retrieval-augmented generation

*Session 3*

- Chunk and index a bilingual rule book
- Write a prompt that cites its sources and refuses when they are silent
- Evaluate retrieval and generation separately

#### [day2#facts-live-in-documents] Facts that change live in documents, not in weights

*Retrieval-augmented generation*

**Customer:** How much does a meter test cost and when is it refunded?

**Model alone (weak example):** A plausible price, from nowhere.

**Model with rule R-07 (good example):** The answer the rule gives, with [R-07] after it.

- The tariff rule book is Latifa's: ten rules, each in English and Arabic.
- No model was trained on it, and it changes every quarter.
- Retrieve the rule, put it in the context, ask for an answer **from it**, with a citation.

#### [day2#ingest-and-answer] Two pipelines: one when the rules change, one for every question

*The shape*

1. **Ingest · when the rule book changes** split each rule into chunks, keep the rule id
2. **Index** normalised words, or embeddings

1. **Answer · for every question** retrieve the top 3 chunks
2. **Generate** answer from them, cite, or refuse
3. **Check** citation present, refusal exact

#### [day2#bm25-with-normalisation] For ten rules, a keyword search with Arabic normalisation finds every answer

*Measured in Lab 2.3*

**12 of 12** answerable questions with the right rule in the top 3 (English, MSA and Gulf)

- **And nothing for Arabizi.** `bkam tif7soon el 3adad?` shares no word with a rule book in Arabic script and English.

- **BM25**: a rare word counts more, a repeated word counts less each time, a long chunk gets no bonus.
- Normalise first, and cut words to six letters so that a word meets its suffixed forms.
- Start with the simplest retriever and an evaluation set. Add embeddings where the set shows a gap.

#### [day2#answer-from-the-rules-only] Answer from the rules, cite the rule, or say the rules are silent

*The generator's contract*

Prompt for Grounded reply, Lab 2.3:

```text
You answer customers of Sawaqi Water Company from the RULES given in the message and from nothing else.
Answer in the language of the question, in one or two sentences,
and cite the rule you used in square brackets, like [R-02].
If the RULES do not answer the question, reply exactly: NOT_IN_RULES
```

Source · Form · Citation · Refusal

The user message holds the three retrieved chunks, each starting with its rule id, then the question.

#### [day2#score-the-two-parts-apart] Score the retriever and the generator apart

*Lab 2.3, step 4*

| Number | Over | The part it measures |
| --- | --- | --- |
| Hit at 3 | 12 answerable questions | the retriever: is the right rule in the context? |
| Cited the right rule | 12 answerable | the generator: did it use that rule? |
| Contains the key fact | 12 answerable | the generator: did it copy the figure correctly? |
| False refusals | 12 answerable | the generator: too cautious |
| Refused | 4 questions the rules do not answer | the generator: not cautious enough when below 4 |

> **One score for the whole pipeline cannot tell you which part to fix.**

#### [day2#locate-the-error] Locate the error before you change anything

*Reading a wrong answer*

- **The rule was not retrieved.** Fix the retriever: wording, normalisation, a second search. The generator never had a chance.
- **Retrieved, answer wrong.** Fix the prompt or the chunk: two tiers in one sentence, a figure copied from the wrong tier.
- **Silent question, answered.** Read which retrieved rule it leaned on. Tighten the refusal clause; test it on more silent questions.

#### [day2#check-grounded] The right rule, the wrong price

*Checkpoint*

**Question:** A customer asks what the second residential tier costs. The top chunk is R-02, which lists all four tiers. The reply says "SAR 1.20 [R-02]"; the rule says SAR 2.50. Where do you look first?

- A. At the generator: the prompt and how the four tiers read in one sentence
- B. At the retriever: return ten chunks instead of three
- C. At the index: replace keywords by embeddings
- D. At the model: fine-tune it on the rule book

#### [day2#lab-2-3] Grounded replies from the rule book

Lab 2.3 · 25 min

Ground replies in the tariff rules with citations and refuse when the rules are silent, scoring retrieval and answers separately on 16 questions.

[D2_3_rag.ipynb](https://sdaiaacademy.github.io/ae-301/labs/D2_3_rag.ipynb)

- [ ] Chunk and index the rule book (6 min)
- [ ] Retrieval: hit at 3 on 16 questions (5 min)
- [ ] Grounded replies with citations (8 min)
- [ ] Score citations and refusals (6 min)
- [ ] ✱ Stretch: Ask in Gulf dialect and in Arabizi: does retrieval still find the rule?

### Session 4: Model selection and evaluation

#### [day2#s4] Model selection and evaluation

*Session 4*

- Compare models on one labelled set by macro F1 and F1 by variety
- Measure latency percentiles and cost per thousand messages
- Defend a choice against a latency budget

#### [day2#choose-on-four-columns] A model is chosen on four columns, and a leaderboard has one

*Model selection*

- **Quality.** Macro F1 on your test set, and by variety.
- **Latency.** Median and 95th percentile, against a budget.
- **Cost.** Per 1,000 messages at your volume.
- **Licence and place.** May you run the weights yourself? Where is the text processed?

> **A public leaderboard** ranks models on someone else's task, in someone else's language, with no price. Use it to choose candidates, never to choose.

#### [day2#cost-is-volume-tokens-price] Cost = messages × tokens per call × the published price

*Cost*

| Model (paid endpoint) | Input, USD per million tokens | Output |
| --- | --- | --- |
| The larger model of the lab | 0.20 | 0.40 |
| The smaller model of the lab | 0.08 | 0.20 |
| The baseline | your own processor |  |

- A routing call is about 200 input tokens and a few output tokens.
- At 0.20 per million: about 0.04 USD per 1,000 messages, 2.40 USD a month at 60,000 messages.
- A **free** endpoint costs nothing and has a daily request limit: price the paid endpoint of the same model.

#### [day2#latency-is-a-distribution] Latency is a distribution: report the median and the 95th percentile

*Latency*

- The **median** (p50) is what a typical message waits.
- The **95th percentile** (p95) is what one message in twenty waits. Budgets are written on it.
- A hosted model's latency includes the network and a queue you do not control.
- Sawaqi's budget: **300 ms at p95** per message.

**4 · 5 ms** the baseline's p50 and p95 on a laptop processor (Lab 1.4)

- **A mean hides the queue.** Nineteen fast calls and one of five seconds average to "fine".

#### [day2#zero-shot-router] Describe the eight departments, ask for one name, parse what comes back

*The prompted model as a classifier*

```python
# file: Lab 2.4
SYSTEM = """You route one customer message of a water
utility to one department. ...
Departments:
- billing: a bill that is too high, a charge twice
- meters: a meter that is broken or reads wrong
...
Reply with the department name only."""

def parse_label(reply):
    low = reply.lower()
    found = [(low.index(d), d)
             for d in DEPARTMENTS if d in low]
    return min(found)[1] if found else None
```

- **No training data**: the descriptions are the whole "training".
- Parse leniently ("Department: leaks"), and count a reply with no department as wrong.
- Temperature 0, a dozen output tokens.

#### [day2#forty-messages-are-few] Forty messages, ten per variety: enough to find failures, not to rank close models

*Measured in Lab 2.4 · the baseline's row*

| Approach | Macro F1 | MSA | Gulf | Arabizi | English |
| --- | --- | --- | --- | --- | --- |
| Baseline, on these 40 | 0.81 | 1.00 | 0.60 | 0.51 | 0.71 |
| Baseline, on all 403 | 0.85 | 0.94 | 0.86 | 0.42 | 0.81 |
| Larger model | measured by your run |  |  |  |  |
| Smaller model |  |  |  |  |  |

- The same baseline scores 0.60 for Gulf on ten messages and 0.86 on 157.
- With ten messages, one error moves a variety by several points.
- Read the errors, then widen the sample before you sign a decision.

#### [day2#cascade-cheap-first] A cascade sends only the unsure messages to the expensive model

*You do not have to choose one*

1. **Baseline** 4 ms, free
2. **Confident?** above a threshold: done
3. **Prompted model** the rest of the messages
4. **Still unsure?** to a person

- Cost and latency are paid only for the share of messages that reach the second step.
- The threshold is chosen on validation data, and its hand-off rate is reported.
- It is a new system: evaluate it as a whole, by variety, like any other row.

#### [day2#write-the-decision] A decision is three sentences someone can disagree with

*The output of model selection*

1. Which approach routes today, and on which numbers.
2. What it costs a month at Sawaqi's volume, and its p95 against the budget.
3. Which messages go to a person until a better model exists.

- **Say what would change it.** "If Arabizi F1 of the prompted model stays above the baseline's on 400 messages, we switch that variety."
- **And what is not in the table.** Licence, where the text is processed, what happens when the model is withdrawn.

#### [day2#check-model-selection] The cheaper model ties on English

*Checkpoint*

**Question:** The smaller model matches the larger one on English and MSA, loses 20 points on Arabizi, and costs less than half. 15% of messages are Arabizi. What do you recommend?

- A. The smaller model for everything, since it costs less than half
- B. The larger model for everything, since it is better somewhere
- C. Either one, since the two models are equivalent
- D. Route by variety: the larger model or a person for Arabizi; confirm on more messages

#### [day2#lab-2-4] Two open-weight models against the baseline

Lab 2.4 · 25 min

Compare two open-weight models as zero-shot routers on 40 labelled messages for F1, latency and cost, next to the Day 1 baseline.

[D2_4_model_selection.ipynb](https://sdaiaacademy.github.io/ae-301/labs/D2_4_model_selection.ipynb)

- [ ] A zero-shot routing prompt and a label parser (6 min)
- [ ] Run both models on 40 messages (7 min)
- [ ] Macro F1 and F1 by variety (6 min)
- [ ] Latency, cost per thousand, decision (6 min)
- [ ] ✱ Stretch: A cascade: the baseline first, the model only when it is unsure

#### [day2#day-2-recap] A prompt asks; code checks; a table decides

*Day 2 recap*

- **Sampling** is a setting: temperature 0 for decisions, and it repeats errors too.
- **Mask** personal fields before text leaves; test the masking.
- **Schema and validators**: valid is not right, so score each field.

- **Grounding**: retrieve, cite, refuse; score retriever and generator apart.
- **Selection**: quality by variety, p95 latency, cost per 1,000, licence.
- **Small samples** find failures; they do not rank close models.

#### [day2#tomorrow-adapt] Adapt: train the third approach, then ship one of the three

*Tomorrow · Day 3*

Split without leakage, fine-tune an encoder for routing and read its confusion matrix, train a low-rank adapter inside a memory budget, and wrap the chosen model in a tested service with a model card.

## Day 3: Adapt

#### [day3#title] Adapt

*SDAIA Academy · Applied Natural Language Processing Engineering*

Day 3

AI Engineering Track · 3 teaching days + capstone day · 4 sessions · 4 labs · AE-301

#### [day3#agenda] Adapt

*Today*

- **Session 1 · Fine-tuning data**: An honest test score needs a split by customer and enough labels in every variety
- **Session 2 · Fine-tuning an encoder**: An encoder plus a small head, trained with a short loop and read through its confusion matrix
- **Session 3 · Low-rank and quantised adapters**: Train a small update instead of every weight, within a memory budget
- **Session 4 · Evaluate and deploy**: One table for every approach, a small tested service and a model card with results by variety

#### [day3#review-day-2] Before we train anything

*Review · Day 2*

**Question:** Ghayth extracts fields with a hosted model. A reply has valid JSON with `"account_number": "3055512345"`. The message, as sent, contained `[ACCOUNT_1]`. What does the service do with this reply?

- A. Accepts it, since it matches the account pattern
- B. Rejects it, since the model only ever saw a placeholder
- C. Accepts it if the number exists in the customer table
- D. Asks a second model whether the number is plausible

### Session 1: Fine-tuning data

#### [day3#s1] Fine-tuning data

*Session 1*

- Decide when fine-tuning is worth it
- Split by customer to prevent leakage
- Check label balance by department and variety

#### [day3#when-to-fine-tune] Fine-tune for a narrow, stable skill you have labels for

*Fine-tuning data*

| The need | Reach for | Because |
| --- | --- | --- |
| A task you can describe, no labels yet | A prompted model | Works today; costs per call |
| Facts that change (the tariff) | Retrieval | Update a file, not a model |
| One decision, thousands of times a day, with labels | A fine-tuned small model | Milliseconds on a processor, no cost per call |
| A large model that must fit small memory | Adapters, quantised weights | Train a fraction of the weights |

> **And always:** the baseline from Day 1 stays in the table. Fine-tuning has to beat it to be worth maintaining.

#### [day3#customers-write-again] Customers write again, and their second message is almost the first

*Measured in Lab 3.1 · 1,106 labelled messages, 345 customers*

**C0009:** Hello, I want a wter connection for a new house in Al-Yasmeen and the meer was replaced last month

**C0009:** I want a water connection for a new house in Al-Yasmeen and the meter was replaced last month. Thanks.

- **274** near-duplicate pairs within customers
- **29%** of labelled messages have at least one near-duplicate

#### [day3#leak-inflates-the-score] A random split tests the model on messages it has nearly seen

*Measured in Lab 3.1 · five seeds each*

| Split | Test messages with a near-copy in training | Macro F1 |
| --- | --- | --- |
| Random | 16% | 0.892 |
| By customer | 0% | 0.859 |

- Same data, same baseline, same sizes: 400 to train, 400 to test.
- The model did not get better. The test got easier.
- The 0.033 of difference is a score no new customer will ever see.
- A larger model memorises better, so it profits **more** from the leak.

#### [day3#split-by-what-repeats] Split by the unit that repeats, and give each part one job

*The rule*

```python
# file: Lab 3.1
ids = df["customer_id"].drop_duplicates() \
        .sample(frac=1, random_state=seed)
sizes = df["customer_id"].value_counts()
side, n_test, n_train = {}, 0, 0
for cid in ids:                 # a customer goes whole
    if n_test < 400:
        side[cid], n_test = "test", n_test + sizes[cid]
    elif n_train < 400:
        side[cid], n_train = "train", n_train + sizes[cid]
```

| Part | Its one job |
| --- | --- |
| Train | fit the weights |
| Validation | choose: epochs, thresholds, prompts, learning rates |
| Test | one final measurement, by variety |

#### [day3#count-labels-by-variety] Count the labels for every department in every variety before training

*Measured in Lab 3.1*

| Department | MSA | Gulf | Arabizi | English |
| --- | --- | --- | --- | --- |
| billing | 21 | 24 | 13 | 7 |
| payments | 18 | *3* | 7 | 9 |
| connections | 6 | 10 | *0* | 9 |
| wastewater | 19 | 18 | *1* | *3* |

- Four cells of 32 have fewer than five training messages in this split.
- Each one is a promise the model cannot keep.
- Three honest answers: label more of exactly that kind, declare the cell unsupported, or send it to a person.

#### [day3#labels-are-decisions] A label is a decision someone made, and some were hard

*Label quality*

**Customer:** My bill is high and I think the meter is broken

**Labelled:** meters

**Customer:** The pressure is weak and I have already paid the last bill

**Labelled:** supply

- The first message is a billing complaint with a meters cause. Either label is defensible; an inconsistent one is not.
- The second carries a clause with another department's words. It is not the complaint.
- Write the rule down, label by it, and expect the model's ceiling to be the labellers' agreement.

#### [day3#data-card] A data card states what the scores were measured on

*The output of this session*

- How many messages in each part, from how many customers.
- How it was split, and the check: customers on both sides, 0.
- Near-duplicates found, and what a random split would have reported.
- Messages per variety; the cells with fewer than five labels.
- Personal data: what is in it, how it is masked. What the data is not for.

- **Why write it.** Every number in the model card is conditional on this page. The capstone's reproducibility criterion asks for it, with the seed.

#### [day3#check-test-set] Six learning rates, one test set

*Checkpoint*

**Question:** A colleague fine-tunes the encoder with six learning rates, scores every run on the test split and reports the best: macro F1 0.66. What is the honest status of that number?

- A. Too few runs: with twelve learning rates it would be reliable
- B. Fine, since the test messages were never used for training
- C. Optimistic: the test split chose the run, so it is no longer unseen
- D. Invalid: all six runs must be repeated on new data

#### [day3#lab-3-1] A split without leakage

Lab 3.1 · 25 min

Split a labelled set by customer to avoid leakage, show how much a random split inflates the score, and check label balance by dialect.

[D3_1_splits.ipynb](https://sdaiaacademy.github.io/ae-301/labs/D3_1_splits.ipynb)

- [ ] Find near-duplicate messages per customer (5 min)
- [ ] Random split against customer split (8 min)
- [ ] Label balance by dialect (6 min)
- [ ] Write the split and a data card (6 min)
- [ ] ✱ Stretch: A split that is grouped by customer and stratified by department

### Session 2: Fine-tuning an encoder

#### [day3#s2] Fine-tuning an encoder

*Session 2*

- Add a classification head to a pretrained encoder
- Fine-tune it with sensible defaults and watch validation
- Read a confusion matrix and name the errors

#### [day3#encoder-plus-head] A pretrained encoder, a mean over real tokens, and eight scores

*Fine-tuning an encoder*

```python
# file: Lab 3.2
class Router(nn.Module):
    def __init__(self, encoder, n_labels):
        super().__init__()
        self.encoder = encoder
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(encoder.cfg.d, n_labels)

    def forward(self, ids):
        mask = ids != 0              # padding out
        h = self.encoder(ids, mask)  # a vector per token
        pooled = (h * mask[..., None]).sum(1) \
                 / mask.sum(1, keepdim=True)
        return self.head(self.drop(pooled))
```

- The **encoder** arrives pretrained: it has read Sawaqi-style messages without labels.
- **Pooling** turns token vectors into one message vector. Padding is left out of the mean.
- The **head** is new: 128 × 8 weights and 8 biases.
- Full fine-tuning trains all 530,440 parameters.

#### [day3#the-training-loop] The training loop is the one you know, with two habits for small data

*Lab 3.2, step 2*

```python
# file: Lab 3.2
def train_epoch(model, opt, x, y):
    model.train()
    perm, total = torch.randperm(len(x)), 0.0
    for s in range(0, len(x), 32):
        idx = perm[s:s + 32]
        loss = F.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    return total / len(x)
```

- **Mask 15% of the input tokens.** Each epoch hides different tokens, as in pretraining. With 400 messages the model would otherwise learn single words by heart.
- **Keep the best validation epoch.** Score the validation split after every epoch and keep those weights.

#### [day3#pick-the-epoch-on-validation] The training loss keeps falling; the validation score stops

*Measured in Lab 3.2 · 40 epochs, about two minutes on a processor*

| Epoch | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Training loss | 1.54 | 0.60 | 0.22 | 0.12 | 0.08 | 0.05 | 0.04 | 0.03 |
| Validation macro F1 | 0.22 | 0.47 | 0.48 | 0.53 | 0.53 | 0.56 | 0.55 | 0.58 |

- From epoch 20 the loss falls by a factor of four and validation moves within five points.
- The loop keeps the weights of the best validation epoch: here epoch 29.
- The test split has not been touched yet.

#### [day3#encoder-against-baseline] The third row of the comparison: here, the baseline wins

*Measured in Lab 3.2 · 403 test messages*

| Approach | Macro F1 | MSA | Gulf | Arabizi | English |
| --- | --- | --- | --- | --- | --- |
| Baseline | 0.85 | 0.94 | 0.86 | 0.42 | 0.81 |
| Fine-tuned course encoder | 0.61 | 0.63 | 0.66 | 0.33 | 0.52 |

- Half a million parameters and 400 labels do not beat character n-grams on short messages.
- That is a result, not a failure of the lab: it is why the baseline row exists.
- What would change it: far more labels, or an encoder pretrained on far more Arabic. The stretch tries the second in Colab.

#### [day3#read-the-confusion-matrix] A confusion matrix says where the messages went

*Measured in Lab 3.2*

| True department | Sent to | Messages |
| --- | --- | --- |
| supply | quality | 13 |
| payments | billing | 11 |
| supply | payments | 8 |
| leaks | payments | 6 |
| billing | payments | 6 |

- Supply and quality share most of their words: water, tap, tank.
- Payments and billing both speak of the bill.
- Supply sent to payments: "…and I have already paid the last bill". The second clause won.
- Read ten wrong messages of the largest cell before you change the model.

#### [day3#model-error-or-label-error] Name the cause before you choose the fix

*After the matrix*

- **The label is arguable.** "High bill, maybe the meter." Decide the rule, relabel consistently, tell the departments.
- **The data is thin.** No Arabizi wastewater messages in training. Label more of that kind, or hand it off.
- **The model is distracted.** A second clause wins. Add examples with such clauses; consider routing on the first sentence.

> **"Train longer" is rarely on this list.** The training loss was already near zero.

#### [day3#check-validation-peak] Validation peaked early

*Checkpoint*

**Question:** You fine-tune for 40 epochs. Validation macro F1 is highest at epoch 12 and slightly lower afterwards, while the training loss falls to 0.01. Which weights do you measure on the test split?

- A. Those of epoch 12, the best validation epoch
- B. Those of epoch 40, where the loss is lowest
- C. Those of whichever epoch scores best on the test split
- D. An average of the weights of all 40 epochs

#### [day3#lab-3-2] Fine-tune the course encoder for routing

Lab 3.2 · 25 min

Fine-tune a small Arabic-capable encoder for routing, score it by variety and read its confusion matrix.

[D3_2_finetune_encoder.ipynb](https://sdaiaacademy.github.io/ae-301/labs/D3_2_finetune_encoder.ipynb)

- [ ] Load the encoder and encode a batch (4 min)
- [ ] A classification head and a training loop (9 min)
- [ ] Macro F1 overall and by variety (5 min)
- [ ] Read the confusion matrix (7 min)
- [ ] ✱ Stretch: Run the same notebook on a published Arabic encoder in Colab with a GPU

### Session 3: Low-rank and quantised adapters

#### [day3#s3] Low-rank and quantised adapters

*Session 3*

- Explain the memory cost of full fine-tuning
- Write a low-rank adapter layer and count trainable parameters
- Compare an adapter run with full fine-tuning on the same split

#### [day3#memory-budget] Training keeps every trainable weight four times

*Low-rank and quantised adapters*

| A weight that… | Bytes during training |
| --- | --- |
| trains (value, gradient, two Adam states) | 16 |
| is frozen, 32-bit | 4 |
| is frozen, 8-bit | 1 |
| is frozen, 4-bit | 0.5 |

| Model (gigabytes for the weights) | Full | Adapter, 32-bit | Adapter, 4-bit |
| --- | --- | --- | --- |
| Encoder, 110 million parameters | 1.8 | 0.4 | 0.1 |
| Decoder, 7 billion parameters | 112 | 28.4 | 4.0 |

Adapters at 0.5% of the weights. Activations come on top and grow with the batch and the length of the text.

#### [day3#low-rank-update] Freeze the weight matrix; learn its change as two thin matrices

*The idea*

1. **Frozen W** 128 × 128 = 16,384 numbers
2. **A: 8 × 128 and B: 128 × 8** 2,048 trainable numbers at rank 8
3. **Output** W x + (alpha / r) · B A x

- The bet: the change a task needs is **low rank**, so two thin matrices can hold it.
- **B starts at zero**: the adapted layer begins exactly as the pretrained one.
- **Rank** sets capacity; **alpha / r** scales the update.
- After training, add B A into W once: no extra latency when serving.

#### [day3#lora-layer-by-hand] The adapter layer is ten lines, and 1.7% of the model trains

*Lab 3.3, steps 1 and 2*

```python
# file: Lab 3.3
class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base, self.scale = base, alpha / r
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(
            torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(
            torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) \
            + (x @ self.A.T @ self.B.T) * self.scale
```

| Course encoder with head | Parameters |
| --- | --- |
| Total | 538,632 |
| Adapters on query and value, 2 layers | 8,192 |
| The new head | 1,032 |
| Trainable | 9,224 (1.7%) |

#### [day3#quantised-frozen-weights] Frozen weights can be stored in fewer bits, at a small error

*Measured in Lab 3.3, step 3*

| Run on the course encoder | Megabytes for training | In a 3 MB budget |
| --- | --- | --- |
| Full fine-tuning | 8.49 | no |
| Adapter, 32-bit frozen weights | 2.27 | yes |
| Adapter, 8-bit frozen weights | 0.68 | yes |

- **Quantising** rounds each weight to one of 255 levels and keeps one scale per matrix.
- The frozen weights of the course encoder move by 2.5% on average.
- Only frozen weights can be held this way: a gradient step is smaller than one level.
- The adapters stay in 32 bits and learn on top of the rounded model.

#### [day3#adapter-vs-full] The adapter trains 1.7% of the weights and gives up 13 points

*Measured in Lab 3.3 · the evidence of LO4*

| Run | Trainable | Macro F1 | MSA | Gulf | Arabizi | English |
| --- | --- | --- | --- | --- | --- | --- |
| Full fine-tuning (supplied) | 530,440 | 0.63 | 0.68 | 0.66 | 0.28 | 0.59 |
| Adapter, rank 8, 8-bit frozen weights | 9,224 | 0.50 | 0.43 | 0.55 | 0.22 | 0.55 |

- Same split, same 40 epochs, same token masking. The logs carry timestamps.
- At this size the saving is 8 megabytes, which nobody needs. The trade is real at 7 billion parameters, where full fine-tuning does not fit at all.

#### [day3#when-adapters-fit] Adapters are for a skill on a large model, not for facts and not for small models

*Using adapters well*

- **Good fit.** A large model that does not fit your memory in full. Several tasks on one frozen model, one small adapter each.
- **Poor fit.** A small model you can fine-tune in full. You pay quality and save nothing you need.
- **Wrong tool.** Facts that change: prices, rules, dates. They belong in the rule book that Day 2 retrieves from.

#### [day3#check-adapters] The tariff changes every quarter

*Checkpoint*

**Question:** Latifa updates the tariff rule book four times a year. A colleague proposes a new low-rank adapter each quarter so that Ghayth "knows" the current prices. What do you answer?

- A. Yes, with a higher rank so that the adapter can hold the prices
- B. Yes, but with full fine-tuning, since adapters lose quality
- C. No: adapters over quantised weights would round the prices
- D. No: keep prices in the rule book and retrieve them; adapt for skills

#### [day3#lab-3-3] A low-rank adapter in a memory budget

Lab 3.3 · 25 min

Train a low-rank adapter in a fixed memory budget and compare it with a supplied full fine-tuning run, with the frozen weights optionally held in 8 bits.

[D3_3_lora.ipynb](https://sdaiaacademy.github.io/ae-301/labs/D3_3_lora.ipynb)

- [ ] A low-rank adapter layer by hand (7 min)
- [ ] Wrap the attention and count parameters (4 min)
- [ ] A memory budget, with 8-bit frozen weights (6 min)
- [ ] Train and compare with full fine-tuning (8 min)
- [ ] ✱ Stretch: Sweep the rank and plot macro F1 against trainable parameters

### Session 4: Evaluate and deploy

#### [day3#s4] Evaluate and deploy

*Session 4*

- Put every approach in one comparison table
- Serve the chosen model behind a small API with a smoke test
- Write a model card with results by variety and a hand-off rule

#### [day3#one-table-every-approach] Every approach in one table, on the same test messages

*The comparison, complete*

| Approach | Macro F1 | MSA | Gulf | Arabizi | English | Latency | Cost per 1,000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.85 | 0.94 | 0.86 | 0.42 | 0.81 | milliseconds | none per call |
| Prompted model | your Lab 2.4 run, widened to the test split | network | per token |  |  |  |  |
| Fine-tuned encoder | 0.63 | 0.68 | 0.66 | 0.28 | 0.59 | milliseconds | none per call |
| Encoder with adapter | 0.50 | 0.43 | 0.55 | 0.22 | 0.55 | milliseconds | none per call |

> **With these rows, the baseline routes.** The open question is Arabizi, and the only row that can answer it is yours.

#### [day3#thin-api-over-a-function] The decision is a function; the API is a thin layer over it

*Deploying*

```python
# file: Lab 3.4
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

@app.post("/route", response_model=Decision)
def route(message: Message) -> Decision:
    return Decision(**decide(message.text))
```

- **FastAPI with Pydantic**: the schema validates what comes in and what goes out, and documents the service.
- A message of 3,000 characters is refused with status 422 before the model runs.
- A **demo page** at the root lets a colleague try a message without a tool. Gradio or Streamlit do the same with more polish.

#### [day3#hand-off-rule] The hand-off rule is part of the model, and it has a price

*Measured in Lab 3.4 · the baseline, as served*

- Fewer than two words: **no request**.
- Best probability below 0.5, or 0.7 for Arabizi: **low confidence**.
- Second department at 60% of the first or more: **two departments**.
- This starting rule is right on every message it keeps, and keeps too few. Tuning it on validation data is the engineer's work; the trade is Hessa's decision.

| Variety | Handed to a person | Accuracy on the rest |
| --- | --- | --- |
| MSA | 19% | 1.00 |
| Gulf | 27% | 1.00 |
| English | 41% | 1.00 |
| Arabizi | 94% | 1.00 |

#### [day3#smoke-test-is-not-quality] A smoke test proves the service is alive and sane at the edges

*Lab 3.4, step 2*

| Case | Must |
| --- | --- |
| 20 real test messages, five per variety | answer 200 with a department from the list |
| An empty message | be handed to a person |
| Punctuation only |  |
| A greeting with no request |  |
| Three problems in one message |  |

- 24 cases, a second to run, on every change.
- It says nothing about quality: that is the table by variety on the test split.
- The course's service passes: 24 of 24, with 18 of the 20 real messages sent to the right department.

#### [day3#latency-through-the-service] Measure latency through the service, not around the model

*Measured in Lab 3.4, step 3*

- **4 ms** the baseline alone, median (Lab 1.4)
- **25 ms** through the service, median
- **31 ms** through the service, 95th percentile; the budget is 300 ms

- The customer waits for validation, extraction, the model, the hand-off rule and the response.
- Report the median and the 95th percentile, on the hardware you will run on, and say which hardware.

#### [day3#model-card-by-variety] A model card says who the model fails, and what happens then

*The output of the course's labs*

1. **What it is for**, and what it is not for.
2. **The served model** and its data card.
3. **Results by language variety**, as served, with the hand-off rate.
4. **The approaches compared**: the table.
5. **Latency**, and on what hardware.
6. **Limits**: varieties below 0.7 macro F1.
7. **The hand-off rule**, in plain words.
8. **Data and privacy**: what is masked, and where.

- **For this service.** "Arabizi messages are routed correctly far less often than the others. Almost all of them go to a person. Do not staff as if they were automated."

#### [day3#check-model-card] One sentence about Arabizi

*Checkpoint*

**Question:** The served router scores macro F1 0.85 overall and 0.42 for Arabizi, and hands 94% of Arabizi messages to a person. Which sentence belongs in the model card?

- A. "Supports Arabic and English with a macro F1 of 0.85."
- B. "Arabizi support will improve in the next version."
- C. "Arabizi: macro F1 0.42; 94% of these messages go to a person."
- D. "Performance varies with out-of-distribution orthography."

#### [day3#lab-3-4] A small service with a model card

Lab 3.4 · 25 min

Wrap the chosen model in a small API with a demo page, pass the smoke test, measure CPU latency and draft the model card.

[D3_4_service.ipynb](https://sdaiaacademy.github.io/ae-301/labs/D3_4_service.ipynb)

- [ ] An API that routes, extracts and hands off (7 min)
- [ ] Pass the smoke test (5 min)
- [ ] CPU latency, median and 95th percentile (5 min)
- [ ] Draft the model card by variety (8 min)
- [ ] ✱ Stretch: Serve the same model with a Gradio or Streamlit demo

#### [day3#day-3-recap] The third row is in, a model is served, and its limits are written down

*Day 3 recap*

- **Split by customer**: a random split reported 0.033 that no new customer sees.
- **Validation chooses**, the test split measures once.
- **Fine-tuned encoder**: 0.63 against the baseline's 0.85 on this data.

- **Adapter**: 1.7% of the weights, 13 points lower; worth it when the model does not fit.
- **Service**: smoke test passed, latency measured through it.
- **Model card**: results by variety and a hand-off rule.

#### [day3#tomorrow-capstone] Capstone: your service, your table, your defence

*Tomorrow · Day 4*

A public repository with the three-way comparison by variety, latency and cost, validated extraction, both adaptation runs, a tested service and a model card. Then eight minutes to defend it, with a batch you have not seen.

## Capstone: Capstone: a routing and extraction service and its defence

#### [day4#title] Capstone: a routing and extraction service and its defence

*SDAIA Academy · Applied Natural Language Processing Engineering*

Capstone

AI Engineering Track · 3 teaching days + capstone day · 8 min talk and demo · 4 min Q&A · AE-301

#### [day4#agenda] Finish, publish, defend

*Day 4*

- **Morning · finish**: Complete the comparison table on the test split, re-run the smoke test, test a clean clone, finish the README and the model card
- **Deadline · publish**: The public repository, with the table, the model card and the timestamped logs committed
- **Afternoon · defend**: Eight minutes each: the table, a re-score on an unseen batch, two errors explained; then four minutes of questions

#### [day4#the-brief] A routing and extraction service, with its evidence

*What you build*

**Customer:** الماي مقطوع من امس في حي الياسمين، رقم الحساب 3055512345

**Your service:** department: supply · confidence: 0.91 · variety: arabic · handoff: false · account_number: 3055512345 · district: yasmeen

- For Sawaqi, or for a comparable text task of your own with synthetic data.
- Routing, compared three ways on one test split.
- Extraction that is masked, validated and scored by field.
- A served model, a smoke test and a model card.

#### [day4#the-comparison-table] Three approaches, by variety, with latency and cost

*The numbers the README reports*

| Approach | Macro F1 | MSA | Gulf | Arabizi | English | Latency p50 / p95 | Cost per 1,000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Classical baseline | … | … | … | … | … | … | … |
| Prompted generative model | … | … | … | … | … | … | … |
| Fine-tuned encoder | … | … | … | … | … | … | … |

- One held-out test split, split by customer, measured once per approach.
- Latency on a processor, and on which one. Cost at the prices you read, with the date.
- A recommendation in three sentences someone can disagree with.

#### [day4#extraction-and-adaptation] Extraction and adaptation are scored on their evidence

*Two criteria, 20 points each*

- **Extraction.**
  - A masking test that passes: no account or phone number leaves.
  - A schema whose validators reject wrong values.
  - Precision and recall for each field, next to a pattern baseline.
  - The log of validation failures, with what the service did.
- **Adaptation.**
  - A full fine-tuning run and a low-rank adapter run, same split.
  - Trainable-parameter counts and the memory budget.
  - Training logs with timestamps; the epoch chosen on validation.
  - Test scores for both, by variety, and which one you would keep.

#### [day4#rubric] Five criteria, 100 points

*How the repository is scored*

| Item | Value |
| --- | --- |
| Three-way comparison | 25% |
| Extraction | 20% |
| Adaptation | 20% |
| Service and model card | 20% |
| Reproducibility | 15% |

> **Pass:** 70% overall, and at least 60% on this rubric. The repository is 60% of the result, the defence 20%, the supervised checks 20%.

#### [day4#defence-format] Eight minutes, then four minutes of questions

*The defence*

- **3 min · the table.** The task, the three rows by variety, and your recommendation.
- **3 min · a re-score.** The assessor gives a labelled batch you have not seen. Score your served model on it, live.
- **2 min · two errors.** Two messages from that batch your model got wrong: the cause of each, and the remedy.

> **AI assistants were allowed in full.** The defence is where you show that the decisions are yours.

#### [day4#what-costs-marks] Four things that cost marks, each checkable in a minute

*Before you publish*

- **No baseline row, or no varieties.** One average hides the group the model fails.
- **The test split chose something.** A threshold, a prompt, an epoch or a learning rate.
- **Unmasked text left the machine.** An account or phone number in a prompt or a log.
- **A number with no log.** Every figure in the README points at a committed file.

#### [day4#running-order] Who defends when

*Running order*

| # | Participant | Repository |
| --- | --- | --- |
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |
| 4 |  |  |
| 5 |  |  |
| 6 |  |  |

#### [day4#final-prep] Ten minutes: clone it clean, run the smoke test

*Final preparation*

Clone your public repository into a new folder, run the smoke test and re-score the example batch. If that fails, the assessor's batch will too.

#### [day4#now-presenting] Defence

*Now defending*

##### Questions

#### [day4#review] What the room learned

*After the defences*

- **The row that won.** How many chose the baseline, the prompted model, the encoder? On what evidence?
- **The weakest variety.** Which variety failed most often, and which hand-off rule handled it best?
- **The re-score gap.** Whose unseen batch came closest to their README, and why?

#### [day4#what-next] The follow-up task, and the AI Engineering Associate badge

*What next*

- **Follow-up task.** Within four weeks, put one text task from your own work through the same table: a baseline, one alternative, results by subgroup, latency and cost.
- **Keep the repository.** This course is one of those that help prepare for the AI Engineering Associate level, and the repository is your record of the work.
