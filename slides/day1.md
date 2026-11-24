---
course: "Applied Natural Language Processing Engineering"
code: "AE-301"
version: "2027.2.0"
deck: "day1"
day: 1
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-301/slides/day1.html"
---

# Applied Natural Language Processing Engineering

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
