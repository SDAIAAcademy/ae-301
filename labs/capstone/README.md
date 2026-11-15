# Capstone starter kit

The capstone of AE-301: **a routing and extraction service for Sawaqi Water Company**, or for a comparable text task of your own with synthetic data. You submit a public GitHub repository and defend it live: eight minutes, then four minutes of questions. The capstone is individual and AI assistants are allowed in full; the defence is where you show that the decisions are yours.

## What the repository must hold

1. **The three-way comparison** in the README: a classical baseline, a prompted generative model and a fine-tuned encoder for routing, on the same held-out test split. For each: macro F1, F1 for each of the four varieties, CPU latency (median and 95th percentile) and cost per thousand messages. Then your recommendation, in three sentences someone can disagree with.
2. **Extraction** of account number, district and complaint type: masked before any hosted model, validated by a schema, with field-level precision and recall, the log of validation failures and a passing masking test.
3. **Adaptation**: a low-rank adapter run and a full fine-tuning run, with training logs, trainable-parameter counts and test scores for both.
4. **The service**: it runs, passes the smoke test, and has a model card with results by variety and the hand-off rule.
5. **Reproducibility**: a clean clone runs with `uv run`; every number in the README is backed by a committed, timestamped log.

## Rubric

| Criterion | Weight |
|---|---|
| Three-way comparison | 25 |
| Extraction | 20 |
| Adaptation | 20 |
| Service and model card | 20 |
| Reproducibility | 15 |

The repository is 60% of the course result, the live defence 20% and the supervised checks 20%. Passing needs 70% overall and at least 60% on this rubric.

## The defence

Eight minutes: your table and your recommendation (three minutes); a **re-score** of your served model on a batch the assessor gives you, run live (three minutes); **two errors** from that batch, explained (two minutes). Then four minutes of questions.

Practise the re-score now:

```bash
cd labs
uv run python capstone/evaluate.py --smoke
uv run python capstone/evaluate.py --batch capstone/batch.example.csv
```

The assessor's batch has the same four columns as `batch.example.csv`: `msg_id`, `text`, `variety`, `department`.

## What is in this folder

| File | Purpose |
|---|---|
| `service.py` | Lab 3.4's service as one file: `/route`, `/health`, a demo page, the hand-off rule. `uv run uvicorn capstone.service:app --port 8000` |
| `evaluate.py` | The smoke test, a re-score on a batch, or the score of a predictions file, by variety. Every run writes a timestamped log into `capstone/logs/` |
| `batch.example.csv` | Twelve labelled messages in the assessor's format (from the validation split) |
| `TEMPLATE_README.md` | The skeleton of your repository's README, with the comparison table and the model card |

## A comparable task of your own

Allowed when it has the same shape: short texts in at least two language varieties, a routing decision with at least five classes, fields to extract, a split by the unit that repeats (the customer, the author), and **synthetic data that you generate with a fixed seed**. No real personal data, in any file or prompt.

## Rules that cost marks when broken

- A comparison without the baseline row, or with one average and no results by variety.
- A test split that was used to choose a threshold, a prompt or a learning rate.
- Text sent to a hosted model without masking.
- A number in the README with no log behind it.
