"""Fine-tunes a small encoder (MiniLM-L6, 22M parameters) as the tier 2 crisis classifier: none / checkin / crisis.

Trains on data/train.jsonl with label smoothing, keeps the epoch with the lowest loss on val + val_real, and saves the
model and tokenizer to model/. Runs on MPS on an 8 GB laptop (small batches).

    ../neuron-explorer/.venv/bin/python train.py
"""
import json
import os
import random

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("CRISIS_BASE", "sentence-transformers/all-MiniLM-L6-v2")
LABELS = ["none", "checkin", "crisis"]
OUT = os.environ.get("CRISIS_MODEL", os.path.join(HERE, "model"))
LR = float(os.environ.get("CRISIS_LR", "5e-5"))


def load(name):
    return [json.loads(s) for s in open(os.path.join(HERE, "data", f"{name}.jsonl"))]


def probs(model, tok, texts, dev, bs=64):
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            x = tok(texts[i:i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
            out.append(torch.softmax(model(**x).logits.float(), -1).cpu())
    return torch.cat(out)


def val_metrics(p, labels):
    y = torch.tensor([LABELS.index(l) for l in labels])
    pred = p.argmax(-1)
    risk = y > 0
    return {"acc": float((pred == y).float().mean()), "risk_recall": float((pred[risk] > 0).float().mean()),
            "crisis_recall": float((pred[y == 2] == 2).float().mean()),
            "false_alarm": float((pred[~risk] > 0).float().mean())}


def main():
    torch.manual_seed(0)
    random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    train, val = load("train"), load("val") + load("val_real")
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=3, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)}).to(dev)
    counts = torch.tensor([sum(r["label"] == l for r in train) for l in LABELS], dtype=torch.float)
    weight = (counts.sum() / (3 * counts)).to(dev)  # balance classes
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    epochs, bs = 3, 16
    steps = epochs * -(-len(train) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    best = None
    for ep in range(epochs):
        model.train()
        random.shuffle(train)
        loss_sum = 0.0
        for i in range(0, len(train), bs):
            b = train[i:i + bs]
            x = tok([r["text"] for r in b], padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
            y = torch.tensor([LABELS.index(r["label"]) for r in b], device=dev)
            loss = torch.nn.functional.cross_entropy(model(**x).logits, y, weight=weight, label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            loss_sum += float(loss) * len(b)
        vp = probs(model, tok, [r["text"] for r in val], dev)
        m = val_metrics(vp, [r["label"] for r in val])
        vy = torch.tensor([LABELS.index(r["label"]) for r in val])
        m["loss"] = float(torch.nn.functional.nll_loss(torch.log(vp + 1e-9), vy))
        score = -m["loss"]
        print(f"epoch {ep + 1}: loss {loss_sum / len(train):.3f} | val {json.dumps({k: round(v, 3) for k, v in m.items()})}",
              flush=True)
        if best is None or score > best:
            best = score
            model.save_pretrained(OUT)
            tok.save_pretrained(OUT)
    print("saved best epoch to", OUT)


if __name__ == "__main__":
    main()
