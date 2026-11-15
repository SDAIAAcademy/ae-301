# Answer key for the trainer

The numbers below are from the course's own runs of the solution notebooks on one laptop processor on 13 November
2026, with the data that `prepare_data.py` writes (seed 301). Labs without a model call give the same numbers on
every machine, up to small differences in training (Labs 3.2 and 3.3). The reference files are in
`data/reference/`; a participant's notebook falls back on them when a step fails, and says so.

**Not run against a model.** The course was built without a working key for a hosted model. The steps that call one
(Lab 1.2 step 2 and the embedding rows after it; Lab 2.1 steps 2 to 4; the model rows of Lab 2.2 steps 3 and 4;
Lab 2.3 steps 3 and 4; the model rows of Lab 2.4) were run with the scripted stand-in in `ghayth.py`
(`ANLP_STANDIN=1`). Their reference files hold stand-in output: they show the shape of a result and say nothing
about any model. Run these labs once with a key before the first cohort and replace the numbers marked "stand-in"
on the Day 1 and Day 2 slides' notes with what you measure.

## What is planted in the data

| Planted | Where | Which lab meets it |
|---|---|---|
| Four varieties with unequal shares (MSA 35%, Gulf 30%, English 20%, Arabizi 15%) | `messages.csv` | every table by variety |
| Arabizi spelled several ways (`mai` or `may`; the digits 3, 5, 7, 9 or the letters a, kh, h, s) | `messages.csv` | 1.1, 1.2, 1.4: Arabizi is the weak column everywhere |
| Typing errors in about one message in ten | `messages.csv` | 1.4, 3.1 |
| A second clause with another department's words in about three messages in ten ("…and I have already paid the last bill") | `messages.csv` | 1.4, 3.2 (confusion matrix: supply sent to payments) |
| One template whose label is arguable: a high bill blamed on the meter, labelled `meters` | `messages.csv` | 3.2, the slide "A label is a decision" |
| Customers who write again: near-copies of their own earlier message | `messages.csv` | 3.1: 274 pairs, 29% of labelled messages |
| Only about 400 labelled training messages; the rest of the archive is the unlabelled `pool` | `split.csv` | 1.4, 3.1 to 3.3 |
| Account numbers in Arabic-Indic digits; messages with a phone number and no account number | `extraction_200.jsonl` | 2.2: the pattern baseline's account-number errors |
| A district name that is also an ordinary word (المنهل: the district Al-Manhal, and a manhole) | `extraction_200.jsonl` | 2.2: district precision below 1 |
| Four questions the rule book does not answer | `rule_questions.jsonl` | 2.3: the reply must refuse |
| Four edge cases: empty, punctuation only, a greeting, three problems at once | `smoke_test.jsonl` | 3.4: each must be handed to a person |
| Diacritics, alef and hamza variants, ta marbuta, Arabic-Indic digits | `parallel_50.csv` | 1.1 step 4 |

## Expected numbers

### Day 1

- **Lab 1.1.** Tokens per message (MSA / Gulf / Arabizi / English): GPT-2 40.1 / 32.7 / 18.8 / 11.8; Qwen3 15.0 /
  14.6 / 19.4 / 11.9; CAMeLBERT 9.7 / 8.7 / 25.5 / 23.6. After normalisation: `[UNK]` tokens 1,052 → 0, distinct
  word forms 2,968 → 1,760. Needs the Hugging Face hub for three tokeniser files (no model weights).
- **Lab 1.2.** Character n-gram recall at 10: MSA 0.329, Gulf 0.288, Arabizi 0.02, English 0.40, all 0.259. The
  embedding row is **stand-in**.
- **Lab 1.3.** Share of attention on later tokens: encoder 0.361, decoder 0. Each model has 529,408 parameters.
- **Lab 1.4.** Baseline macro F1 0.847 (MSA 0.938, Gulf 0.856, Arabizi 0.420, English 0.808), accuracy 0.861, on 403
  test messages. Latency is the laptop's own: the course measured a median of 4.1 ms and 4.7 ms at the 95th
  percentile.

### Day 2

- **Lab 2.1.** Step 1 (course decoder, no hosted call): probability of the top token 0.996 / 0.631 / 0.335 / 0.108
  at temperatures 0.2 / 0.7 / 1.0 / 1.5. Steps 2 to 4 are **stand-in**.
- **Lab 2.2.** Masking test: 200 of 200 messages leave without an account or phone number. Pattern baseline:
  account number precision 0.693, recall 0.764; district 0.966 / 0.966; complaint type 0.575. Model rows are
  **stand-in**.
- **Lab 2.3.** 20 chunks; retrieval hit at 3 on the 12 answerable questions: 12 of 12 (character n-grams, no hosted
  call). In the stretch, the Arabizi question finds nothing. Steps 3 and 4 are **stand-in**.
- **Lab 2.4.** Baseline on the 40 messages: macro F1 0.810 (1.00 / 0.60 / 0.505 / 0.711). Model rows are
  **stand-in**. Prices were read on 13 November 2026 and are in `data/reference/prices.json`; read them again.

### Day 3 (no hosted model anywhere)

- **Lab 3.1.** 1,106 labelled messages from 345 customers; 274 near-duplicate pairs touching 321 messages (29%).
  Random split: 16.3% of test messages have a near-copy in training, macro F1 0.892; customer split: 0%, 0.859
  (means of five seeds).
- **Lab 3.2.** Fine-tuned course encoder: macro F1 about 0.61 (MSA 0.63, Gulf 0.66, Arabizi 0.33, English 0.52);
  best validation epoch near 29 of 40. Expect ±0.03 between machines. Largest confusions: supply → quality 13,
  payments → billing 11, supply → payments 8. The supplied full run in `data/reference/full_ft_log.json` scored
  0.629.
- **Lab 3.3.** 538,632 parameters, 9,224 trainable (1.7%). Training memory for the weights: 8.49 MB full, 2.27 MB
  adapter, 0.68 MB adapter over 8-bit frozen weights; budget 3 MB. Mean rounding error of 8-bit weights 2.45%.
  Adapter of rank 8: macro F1 about 0.50 (0.43 / 0.55 / 0.22 / 0.55).
- **Lab 3.4.** Smoke test passes (24 cases; 18 of the 20 real messages are sent to the right department). Baseline
  as served: handed to a person MSA 19%, Gulf 27%, Arabizi 94%, English 41%; accuracy on the messages it keeps
  1.00. Latency through the service: the course measured a median of 25 ms and 31 ms at the 95th percentile (43 and 58 in another run, with the machine busy).

## The result to be ready for

The baseline beats both neural runs on this data. This is deliberate honesty, not a defect of the labs: the course
encoder has half a million parameters and was pretrained on synthetic text only. The stretch of Lab 3.2 runs the
same notebook on a published Arabic encoder in Colab; the course did not run it, so do not promise its result.
