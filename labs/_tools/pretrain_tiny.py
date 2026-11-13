"""Train the course's supplied checkpoints (run once by the course team; the results are committed).

    cd labs
    uv run python _tools/pretrain_tiny.py

Writes into labs/models/:
    ghayth-bpe.json              a byte-pair tokeniser with 2,000 entries, trained on the train and pool text
    ghayth-enc-tiny.pt           the encoder, pretrained by masked-token prediction
    ghayth-dec-tiny.pt           the decoder, pretrained by next-token prediction
    ghayth-enc-routing-full.pt   the encoder fine-tuned for routing, every weight trained (Lab 3.3's comparison)
and data/reference/full_ft_log.json, the log of that full fine-tuning run.

The tokeniser and the decoder read the text of the train and pool messages, without labels. The encoder
also reads 10,000 further synthetic messages (other customers, other seeds; generated here, not shipped),
which stand for the unlabelled archive a utility already holds. Fine-tuning reads the 400 labelled train
messages. The course's validation and test messages are never read. Fixed seeds; on a
laptop processor the whole script takes about half an hour; `pretrain_tiny.py encoder` (or `decoder`,
`finetune`) runs one part.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers  # noqa: E402

import ghayth as G  # noqa: E402
import tinytf as T  # noqa: E402


def corpus() -> list[str]:
    texts = list(G.messages("train")["text"]) + list(G.messages("pool")["text"])  # text only: no labels
    for body in G.rules().values():
        texts += [line for line in body.splitlines() if line.strip()]
    return texts


def older_archive(n_batches: int = 5) -> list[str]:
    """10,000 further synthetic messages for pretraining the encoder: other customers, other seeds, text only.

    They stand for the years of unlabelled messages a utility already holds. They are generated here and
    not shipped; none of them is one of the course's 2,000 messages.
    """
    import random

    import prepare_data as P
    texts = []
    for b in range(n_batches):
        rng = random.Random(9000 + b)
        texts += [m["text"] for m in P.make_messages(P.make_customers(rng), rng)]
    return texts


def train_tokenizer(texts: list[str]) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation(),
                                                 pre_tokenizers.Digits(individual_digits=True)])
    tok.decoder = decoders.BPEDecoder()
    trainer = trainers.BpeTrainer(vocab_size=2000, special_tokens=T.SPECIALS, min_frequency=2, show_progress=False)
    tok.train_from_iterator(texts, trainer)
    return tok


def pretrain(texts, tok, causal: bool, epochs: int, label: str, batch: int = 32) -> T.TinyTransformer:
    torch.manual_seed(301)
    model = T.TinyTransformer(T.Config(vocab=tok.get_vocab_size(), causal=causal))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    ids = T.encode(tok, texts, cls=not causal)
    mask_id = tok.token_to_id("[MASK]")
    n = len(ids)
    t0 = time.perf_counter()
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        model.train()
        for s in range(0, n, batch):
            rows = ids[perm[s:s + batch]]
            if causal:
                x, y = rows[:, :-1], rows[:, 1:].clone()
                y[y == 0] = -100
            else:
                real = (rows != 0) & (rows != tok.token_to_id("[CLS]"))
                pick = real & (torch.rand(rows.shape) < 0.15)
                x = rows.clone()
                x[pick] = mask_id
                y = torch.full_like(rows, -100)
                y[pick] = rows[pick]
            logits = model.lm_logits(model(x))
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(rows)
        if ep % 2 == 1 or ep == epochs - 1:
            print(f"  {label} epoch {ep + 1}: loss {total / n:.3f} ({time.perf_counter() - t0:.0f} s)")
    return model.eval()


def full_finetune(tok, encoder_path: Path, epochs: int = 40) -> dict:
    """Every weight trained, 15% of the input tokens masked in each batch, the best validation epoch kept:
    the run Lab 3.3 compares its adapter with."""
    import copy
    mask_id = tok.token_to_id("[MASK]")
    torch.manual_seed(301)
    labels = G.DEPARTMENTS
    train, val, test = G.messages("train"), G.messages("val"), G.messages("test")
    enc = lambda df: T.encode(tok, list(df["text"]))
    lab = lambda df: torch.tensor([labels.index(d) for d in df["department"]])
    xtr, ytr, xva, yva, xte, yte = enc(train), lab(train), enc(val), lab(val), enc(test), lab(test)
    model = T.Classifier(T.load(encoder_path), len(labels))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    log, t0, best, best_state = [], time.perf_counter(), -1.0, None

    def macro(x, y):
        model.eval()
        with torch.no_grad():
            pred = model(x).argmax(-1)
        model.train()
        return G.by_variety([labels[i] for i in y], [labels[i] for i in pred], [None] * len(y)).iloc[0]["macro_f1"], pred

    for ep in range(epochs):
        perm = torch.randperm(len(xtr))
        total = 0.0
        for s in range(0, len(xtr), 32):
            idx = perm[s:s + 32]
            loss = F.cross_entropy(model(T.mask_tokens(xtr[idx], mask_id)), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        f1, _ = macro(xva, yva)
        log.append({"epoch": ep + 1, "train_loss": round(total / len(xtr), 4), "val_macro_f1": float(f1),
                    "elapsed_s": round(time.perf_counter() - t0, 1)})
        if f1 > best:
            best, best_state = float(f1), copy.deepcopy(model.state_dict())
        if ep % 5 == 4:
            print(f"  full fine-tuning epoch {ep + 1}: {log[-1]}")
    model.load_state_dict(best_state)
    _, pred = macro(xte, yte)
    table = G.by_variety([labels[i] for i in yte], [labels[i] for i in pred], test["variety"])
    params = T.count_params(model)
    T.save(model, G.MODELS / "ghayth-enc-routing-full.pt", labels=",".join(labels))
    return {"method": "full fine-tuning", "trainable_params": params, "total_params": params, "epochs": epochs,
            "lr": 1e-3, "batch": 32, "token_masking": 0.15, "kept_epoch": max(log, key=lambda e: e["val_macro_f1"])["epoch"], "log": log, "test": table.to_dict(orient="records"),
            "predictions": [labels[i] for i in pred]}


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else "all"  # all | encoder | decoder | finetune
    G.MODELS.mkdir(exist_ok=True)
    texts = corpus()
    if only == "all":
        tok = train_tokenizer(texts)
        tok.save(str(G.MODELS / "ghayth-bpe.json"))
        print(f"tokeniser: {tok.get_vocab_size()} entries from {len(texts)} lines")
    tok = T.load_tokenizer()
    if only in ("all", "encoder"):
        enc = pretrain(texts + older_archive(), tok, causal=False, epochs=12, label="encoder", batch=64)
        T.save(enc, G.MODELS / "ghayth-enc-tiny.pt")
        print(f"encoder parameters: {T.count_params(enc):,}")
    if only in ("all", "decoder"):
        dec = pretrain(texts, tok, causal=True, epochs=40, label="decoder")
        T.save(dec, G.MODELS / "ghayth-dec-tiny.pt")
    if only in ("all", "encoder", "finetune"):
        run = full_finetune(tok, G.MODELS / "ghayth-enc-tiny.pt")
        G.REFERENCE.mkdir(parents=True, exist_ok=True)
        (G.REFERENCE / "full_ft_log.json").write_text(json.dumps(run, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(pd_table(run["test"]))


def pd_table(rows) -> str:
    return "\n".join(f"  {r['variety']:8s} n={r['n']:4d} macro F1 {r['macro_f1']:.3f} accuracy {r['accuracy']:.3f}" for r in rows)


if __name__ == "__main__":
    main()
