---
course: "Applied Natural Language Processing Engineering"
code: "AE-301"
version: "2027.1.0"
deck: "day2"
day: 2
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-301/slides/day2.html"
---

# Applied Natural Language Processing Engineering

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
