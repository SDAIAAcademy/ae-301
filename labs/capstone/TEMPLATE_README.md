# Ghayth routing and extraction service

**Author:** your name · **Task:** Sawaqi message routing and extraction (or your comparable task) · **Run:** `uv run uvicorn capstone.service:app --port 8000`

## The task

Who has this problem and what a wrong answer costs them, in three sentences. **The service takes** a customer message **and returns** a department, an account number, a district, a complaint type and a hand-off decision.

## Data

| Split | Messages | MSA | Gulf | Arabizi | English | How it was made |
|---|---|---|---|---|---|---|
| train | | | | | | |
| validation | | | | | | |
| test | | | | | | |

Split by customer; customers on both sides: 0 (how you checked). Near-duplicates found: how many. Department and variety pairs with fewer than 5 training messages: which. All data is synthetic, generated with seed: which.

## The three-way comparison (held-out test split)

| Approach | Macro F1 | MSA | Gulf | Arabizi | English | CPU latency p50 / p95 | Cost per 1,000 messages |
|---|---|---|---|---|---|---|---|
| Classical baseline | | | | | | | |
| Prompted generative model (name, date) | | | | | | | |
| Fine-tuned encoder | | | | | | | |

Logs: the files in `capstone/logs/` behind each row.

**Recommendation.** Which approach routes, on which numbers. What it costs a month at the stated volume and its p95 against the budget. Which messages go to a person, and what result would change this recommendation.

## Extraction

| Field | Precision | Recall | Baseline precision | Baseline recall |
|---|---|---|---|---|
| Account number | | | | |
| District | | | | |
| Complaint type | | | | |

Masking test: passes on how many messages. Validation failures: how many on the first attempt, how many after one repair, the three most frequent kinds (file: the failure log).

## Adaptation

| Run | Trainable parameters | Memory for training | Test macro F1 | Arabizi F1 |
|---|---|---|---|---|
| Full fine-tuning | | | | |
| Low-rank adapter (rank, where) | | | | |

Training logs with timestamps: which files. What the adapter gave up, for what saving.

## Model card

- **Intended use, and what it is not for.**
- **Results by language variety**, as served, with the hand-off rate for each.
- **Limits.** Varieties and department pairs the model must not be trusted on.
- **Hand-off rule.** When a message goes to a person, in one sentence a contact-centre lead can act on.
- **Data and privacy.** Synthetic data; personal fields masked before any hosted model.

## Reproduce

```bash
git clone <this repository> && cd <folder>/labs
uv sync
uv run python capstone/evaluate.py --smoke
uv run python capstone/evaluate.py --batch capstone/batch.example.csv
```

## Use of AI assistants

Which assistants you used and for what. Which decisions in this README are yours, and one place where you overruled a suggestion.
