"""Builds train/val splits for the tier 2 classifier from gen_data.py's synthetic messages and label_real.py's real ones.

Synthetic: keeps messages whose two labels (the level asked for at generation, the independent label pass) agree,
and splits by generation batch so near-identical messages from one batch never land on both sides (val.jsonl).
Real: keeps messages both labelling models put at the same level, 15% held out at random (val_real.jsonl).
Both: drop exact and near duplicates, and anything near a test message (tests.py) or a C-SSRS post (eval_cssrs.py).

    python3 build_data.py
"""
import collections
import json
import os
import random
import re

import eval_cssrs as cssrs
from tests import sets

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def toks(s):
    return set(re.findall(r"[a-z']+", s.lower()))


def near(a, b, thr=0.7):
    return len(a & b) / max(1, len(a | b)) >= thr


def real_rows(near_test):
    """Real messages (long and short) both labelling models agree on, minus anything near a test message."""
    a, b = {}, {}
    for kind in ("real", "real_short"):
        for tag, d in (("a", a), ("b", b)):
            path = os.path.join(DATA, f"{kind}_labels_{tag}.jsonl")
            if os.path.exists(path):
                d.update({(r := json.loads(s))["key"]: r for s in open(path)})
    out, n_near = [], 0
    for text, ra in a.items():
        rb = b.get(text)
        if not rb or "level" not in ra or ra.get("level") != rb.get("level"):
            continue
        if near_test(toks(text)):
            n_near += 1
            continue
        out.append({"text": text, "label": ra["level"], "subcat": ra["subreddit"] or "goemotions",
                    "group": f"real|{ra['source']}"})
    print(f"real: {len(out)} kept, {n_near} near a test message")
    return out


def main():
    rows = [json.loads(s) for s in open(os.path.join(DATA, "labels.jsonl"))]
    rows = [r for r in rows if r.get("level") == r["gen_level"]]
    test_toks = [toks(t) for s in sets().values() for t, _ in s]
    if os.path.exists(cssrs.PATH):
        test_toks += [toks(p) for _, _, ps in cssrs.users() for p in ps]
    by_len = collections.defaultdict(list)  # near() needs comparable sizes, so bucket test token sets by length
    for x in test_toks:
        by_len[len(x) // 10].append(x)

    def near_test(t):
        b = len(t) // 10
        return any(near(t, x) for k in (b - 1, b, b + 1) for x in by_len.get(k, ()))
    kept, seen, dropped = [], [], collections.Counter()
    for r in rows:
        t = toks(r["key"])
        if near_test(t):
            dropped["near_test"] += 1
            continue
        if any(near(t, x, 0.85) for x in seen):
            dropped["near_dup"] += 1
            continue
        seen.append(t)
        kept.append({"text": r["key"], "label": r["level"], "subcat": r["subcat"], "group": f"{r['subcat']}|{r['style']}"})
    groups = sorted({r["group"] for r in kept})
    random.Random(0).shuffle(groups)
    val_groups = set()
    by_sub = collections.defaultdict(list)
    for g in groups:
        by_sub[g.split("|")[0]].append(g)
    for gs in by_sub.values():  # one batch per subcategory goes to val
        val_groups.add(gs[0])
    split = {"train": [r for r in kept if r["group"] not in val_groups],
             "val": [r for r in kept if r["group"] in val_groups]}
    for name, rs in split.items():
        with open(os.path.join(DATA, f"{name}.jsonl"), "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        print(name, len(rs), dict(collections.Counter(r["label"] for r in rs)))
    real = real_rows(near_test)
    rng = random.Random(1)
    rng.shuffle(real)
    n_val = len(real) * 15 // 100
    split["train"] += real[n_val:]
    split["val_real"] = real[:n_val]
    with open(os.path.join(DATA, "train.jsonl"), "w") as f:
        for r in split["train"]:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(DATA, "val_real.jsonl"), "w") as f:
        for r in split["val_real"]:
            f.write(json.dumps(r) + "\n")
    for name in ("train", "val_real"):
        print(name, "with real", len(split[name]), dict(collections.Counter(r["label"] for r in split[name])))
    print("dropped", dict(dropped), "| label disagreements", len([1 for s in open(os.path.join(DATA, "labels.jsonl"))]) - len(rows))


if __name__ == "__main__":
    main()
