"""Samples real messages for classifier training from two public datasets, keeping only the text (no usernames):

- Reddit Mental Health Dataset (Low et al. 2020, PDDL, https://zenodo.org/records/3941387; re-use subject to
  Reddit API terms): 2019 posts from mental-health and everyday subreddits, streamed from Zenodo and reservoir-
  sampled per subreddit, so the 750 MB of CSVs never land on disk.
- GoEmotions (Google, Apache 2.0, https://huggingface.co/datasets/google-research-datasets/go_emotions): short
  Reddit comments, oversampling the sad and fearful emotions.

Writes external/real_sample.jsonl: {"text", "source", "subreddit" or "emotions"}. Labels come later (label_real.py).
With --short, writes external/real_short.jsonl instead: the opening one or two sentences of other posts (at most 30
words), and more GoEmotions comments, because app messages are short and long posts taught the classifier that
plain sadness looks like risk. Messages already in real_sample.jsonl are skipped.

    python3 -I fetch_real.py [--short]
"""
import csv
import io
import json
import os
import random
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "external", "real_sample.jsonl")
ZENODO = "https://zenodo.org/api/records/3941387/files/{}_2019_features_tfidf_256.csv/content"
PER_SUB = {"suicidewatch": 900, "depression": 500, "lonely": 300, "EDAnonymous": 300, "addiction": 200,
           "alcoholism": 200, "anxiety": 250, "bpd": 250, "ptsd": 200, "mentalhealth": 250, "socialanxiety": 150,
           "healthanxiety": 150, "relationships": 300, "divorce": 200, "parenting": 200, "personalfinance": 200,
           "fitness": 200, "legaladvice": 150, "teaching": 100, "meditation": 150, "jokes": 100}
GOEMO = ("https://huggingface.co/datasets/google-research-datasets/go_emotions/resolve/main/simplified/"
         "train-00000-of-00001.parquet")
EMOTIONS = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
            "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness",
            "surprise", "neutral"]
DISTRESS = {"sadness", "grief", "fear", "nervousness", "remorse", "disappointment", "embarrassment", "anger"}


def clean(text):
    text = " ".join(text.split())
    return text[:1000]


def reddit(rng):
    csv.field_size_limit(10 ** 8)
    for sub, k in PER_SUB.items():
        keep, seen = [], 0
        with urllib.request.urlopen(ZENODO.format(sub), timeout=300) as r:
            for row in csv.DictReader(io.TextIOWrapper(r, encoding="utf-8", errors="replace")):
                post = row.get("post", "")
                if not 20 <= len(post) <= 3000 or post in ("[removed]", "[deleted]"):
                    continue
                seen += 1
                if len(keep) < k:
                    keep.append(post)
                elif (j := rng.randrange(seen)) < k:
                    keep[j] = post
        print(f"{sub}: kept {len(keep)} of {seen}", flush=True)
        for p in keep:
            yield {"text": clean(p), "source": "low2020", "subreddit": sub}


def goemotions(rng, n_distress=2000, n_other=1500):
    import pyarrow.parquet as pq  # only needed here
    data = urllib.request.urlopen(GOEMO, timeout=300).read()
    t = pq.read_table(io.BytesIO(data)).to_pydict()
    rows = [(txt, [EMOTIONS[i] for i in labs]) for txt, labs in zip(t["text"], t["labels"]) if len(txt) >= 12]
    distress = [r for r in rows if DISTRESS & set(r[1])]
    other = [r for r in rows if not DISTRESS & set(r[1])]
    print(f"goemotions: {len(distress)} distress, {len(other)} other", flush=True)
    for txt, emo in rng.sample(distress, n_distress) + rng.sample(other, n_other):
        # GoEmotions masks names as [NAME]; keep the placeholder, it carries no identity
        yield {"text": clean(txt), "source": "goemotions", "emotions": emo}


def opening(post, max_words=30):
    """The first one or two sentences of a post, if they make a short message on their own."""
    sents = re.split(r"(?<=[.!?])\s+", " ".join(post.split()))
    out = sents[0]
    if len(sents) > 1 and len((out + " " + sents[1]).split()) <= max_words:
        out += " " + sents[1]
    return out if 4 <= len(out.split()) <= max_words else None


def short(rng, per_sub=150, n_distress=2500, n_other=1500):
    seen = set()
    if os.path.exists(OUT):
        seen = {json.loads(s)["text"] for s in open(OUT)}
    rows = []
    for r in reddit(rng):
        o = opening(r["text"])
        if o and o not in seen:
            seen.add(o)
            rows.append({**r, "text": o})
    by_sub = {}
    for r in rows:
        by_sub.setdefault(r["subreddit"], []).append(r)
    out = [x for rs in by_sub.values() for x in rng.sample(rs, min(per_sub, len(rs)))]
    out += [g for g in goemotions(rng, n_distress, n_other) if g["text"] not in seen]
    return out


def main():
    if "--short" in sys.argv:
        rng = random.Random(7)
        global PER_SUB
        PER_SUB = {k: 600 for k in PER_SUB}  # sample more posts, keep the openings that are short enough
        rows = short(rng)
        path = os.path.join(HERE, "external", "real_short.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("->", path, len(rows), "rows")
        return
    rng = random.Random(0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = 0
    with open(OUT, "w") as f:
        for row in list(reddit(rng)) + list(goemotions(rng)):
            f.write(json.dumps(row) + "\n")
            n += 1
    print("->", OUT, n, "rows")


if __name__ == "__main__":
    main()
