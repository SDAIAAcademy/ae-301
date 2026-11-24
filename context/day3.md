---
course: "Applied Natural Language Processing Engineering"
code: "AE-301"
version: "2027.2.0"
deck: "day3"
day: 3
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-301/slides/day3.html"
---

# Applied Natural Language Processing Engineering

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
- Read agreement between two labellers and revise the guideline before labelling more

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
- The model's ceiling is the labellers' agreement: measure it before training.

#### [day3#agreement-table] Two labellers, 60 messages: 80% agree, and one cell holds most of the rest

*Label quality · complaint types in billing and payments*

| C \ D | high bill | duplicate charge | payment failed |
| --- | --- | --- | --- |
| high bill | 27 | 2 | 0 |
| duplicate charge | 1 | 11 | 1 |
| payment failed | 1 | *7* | 10 |

- **Observed** agreement: the diagonal, 48 of 60, 0.80.
- **Chance** agreement: labelling at their own rates without reading, 0.36. Half the messages are high bills.
- **Kappa** = (0.80 − 0.36) / (1 − 0.36) = 0.69.
- 7 of the 12 splits sit in one cell. That is a rule missing, not careless people.

#### [day3#fix-the-guideline] Fix the guideline, label the same sample again, then train

*Label quality · what to do with low agreement*

- **The cell's messages.** "The amount left my card but the bill still shows unpaid." C read a failed payment; D read money taken twice.
- **Guideline v2 adds one rule.** Money left, bill unpaid: payment failed. The same bill paid twice: duplicate charge. One example of each.

| Tempting move | Why it fails |
| --- | --- |
| Train anyway | The model learns both readings |
| Drop the split messages | The hard cases leave the test set, not production |
| A third labeller breaks ties | The rule stays unwritten; the next labeller splits again |

> **Then measure again.** The same 60, labelled blind by v2: 0.93 observed, kappa 0.89.

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
