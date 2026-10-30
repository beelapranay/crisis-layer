"""Builds the v2 splits from relabel.py's labels: a message is kept where both labelling models agree.

- test_real.jsonl: the frozen real test split (data/test_real_texts.json), for a final score only
- train.jsonl / val.jsonl / val_real.jsonl: as build_data.py (synthetic val by generation batch, real val 15% at
  random), minus anything near a test prompt or a C-SSRS post, and synthetic near-duplicates

The v1 splits are moved to data/v1/ first.

    python3 build_v2.py
"""
import collections
import json
import os
import random
import shutil

import eval_cssrs as cssrs
from build_data import near, toks
from tests import sets

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def write(name, rows):
    with open(os.path.join(DATA, f"{name}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{name:10} {len(rows):6}  {dict(collections.Counter(r['label'] for r in rows))}")


def main():
    v1 = os.path.join(DATA, "v1")
    os.makedirs(v1, exist_ok=True)
    for n in ("train", "val", "val_real"):
        p = os.path.join(DATA, f"{n}.jsonl")
        if os.path.exists(p) and not os.path.exists(os.path.join(v1, f"{n}.jsonl")):
            shutil.move(p, v1)

    a = {(r := json.loads(s))["key"]: r for s in open(os.path.join(DATA, "v2_labels_a.jsonl"))}
    b = {(r := json.loads(s))["key"]: r for s in open(os.path.join(DATA, "v2_labels_b.jsonl"))}
    rows, disagree = [], 0
    for t, ra in a.items():
        rb = b.get(t)
        if not rb or "level" not in ra or ra.get("level") != rb.get("level"):
            disagree += 1
            continue
        rows.append({"text": t, "label": ra["level"], "kind": ra["kind"], "subcat": ra["subcat"],
                     "group": ra["group"]})
    print(f"{len(rows)} agreed, {disagree} disagreed or missing")

    test_texts = set(json.load(open(os.path.join(DATA, "test_real_texts.json"))))
    write("test_real", [r for r in rows if r["text"] in test_texts])
    rows = [r for r in rows if r["text"] not in test_texts]

    by_len = collections.defaultdict(list)
    for x in [toks(t) for s in sets().values() for t, _ in s] + [toks(p) for _, _, ps in cssrs.users() for p in ps]:
        by_len[len(x) // 10].append(x)

    def near_test(t):
        k = len(t) // 10
        return any(near(t, x) for j in (k - 1, k, k + 1) for x in by_len.get(j, ()))

    dropped = collections.Counter()
    synth, real, seen = [], [], []
    for r in rows:
        t = toks(r["text"])
        if near_test(t):
            dropped["near_test"] += 1
        elif r["kind"] == "synthetic":
            if any(near(t, x, 0.85) for x in seen):
                dropped["near_dup"] += 1
                continue
            seen.append(t)
            synth.append(r)
        else:
            real.append(r)
    print("dropped", dict(dropped))

    groups = collections.defaultdict(list)
    for g in sorted({r["group"] for r in synth}):
        groups[g.split("|")[0]].append(g)
    rng = random.Random(0)
    val_groups = {rng.choice(gs) for gs in groups.values()}
    rng = random.Random(1)
    rng.shuffle(real)
    n_val = len(real) * 15 // 100
    write("train", [r for r in synth if r["group"] not in val_groups] + real[n_val:])
    write("val", [r for r in synth if r["group"] in val_groups])
    write("val_real", real[:n_val])


if __name__ == "__main__":
    main()
