---
course: "Applied Natural Language Processing Engineering"
code: "AE-301"
version: "2027.2.0"
deck: "day4"
day: 4
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-301/slides/day4.html"
---

# Applied Natural Language Processing Engineering

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
