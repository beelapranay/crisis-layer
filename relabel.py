"""Relabels every training message (synthetic and real) under the v2 level definitions, with two different LLMs.

v1 listed "hopelessness" as a check-in sign and told the labeller to pick the higher level when unsure, so it put
about one in five plainly sad but safe messages at check-in, and the classifiers learned "sad = check-in". v2 makes
sadness, numbness, low motivation, guilt and other depression symptoms "none" unless they come with a sign of
danger, and asks for the best-fitting level instead of the higher one.

Before labelling, it freezes a test split of real messages (data/test_real_texts.json), drawn once with a fixed
seed and never used for training or thresholds.

    python3 relabel.py
"""
import json
import os
import random
import sys

from gen_data import LABEL_SCHEMA, call, key, run

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
LABELERS = {"a": "deepseek/deepseek-v4.1-flash-1017", "b": "deepseek/deepseek-v4-flash"}
N_TEST = 1000

LEVELS_V2 = """- none: no sign of danger to anyone's life or safety. This includes sadness, low mood, numbness, low
  motivation, loneliness, grief, guilt, shame, self-criticism, feeling like a failure, stress, anger, anxiety,
  venting, and depression symptoms, however strong, as long as there is no mention or hint of death, dying, not
  existing, self-harm or being unsafe. Also ordinary health, diet or fitness questions, figures of speech ("this
  traffic is killing me"), talk about fiction or news, and past struggles when the person is now safe.
- checkin: possible danger, worth a gentle question about safety: a passive wish to die, not wake up, disappear or
  stop existing; feeling others would be better off without them; saying there is no reason to live, or that they
  can't go on in a way that may mean ending their life; preoccupation with death or their own funeral; vague hints
  of self-harm; heavy drinking or drug use to cope; disordered eating (restricting, purging, punishing themselves for
  eating); stopping psychiatric medication on their own; wanting to hurt or punish themselves.
- crisis: clear danger now: suicidal intent, a plan, access to means, a recent or ongoing attempt or overdose, active
  self-harm, being abused or in danger, voices telling them to harm themselves, intent to harm someone else, acute
  eating-disorder danger, or asking for methods, lethal amounts or ways to hide self-harm or not eating."""

PROMPT = ("Classify this message sent to a mental wellbeing chatbot into one level, judging the danger to the person "
          "writing it (or to someone they might harm). Someone describing another person's crisis, giving support, or "
          "talking about the past while now safe is not in danger themselves.\n\n{levels}\n\nPick the level that fits "
          "best. Sadness or distress alone, however intense, is 'none' unless it comes with one of the check-in or "
          "crisis signs.\n\nMessage:\n<<<{text}>>>")


def messages():
    """{text: metadata} for every message labelled so far: synthetic (labels.jsonl) and real (real*_labels_a)."""
    out = {}
    for s in open(os.path.join(DATA, "labels.jsonl")):
        r = json.loads(s)
        out.setdefault(r["key"], {"kind": "synthetic", "subcat": r["subcat"], "group": f"{r['subcat']}|{r['style']}"})
    for name in ("real_labels_a.jsonl", "real_short_labels_a.jsonl"):
        for s in open(os.path.join(DATA, name)):
            r = json.loads(s)
            out.setdefault(r["key"], {"kind": "real", "subcat": r["subreddit"] or "goemotions",
                                      "group": f"real|{r['source']}"})
    return out


def main():
    msgs = messages()
    test_path = os.path.join(DATA, "test_real_texts.json")
    if not os.path.exists(test_path):  # frozen once
        real = sorted(t for t, m in msgs.items() if m["kind"] == "real")
        json.dump(random.Random(2026).sample(real, N_TEST), open(test_path, "w"))
    print(f"{len(msgs)} messages, test split frozen in {test_path}", file=sys.stderr)
    k = key()
    for tag, model in LABELERS.items():
        tasks = [{"key": t, **m, "labeler": model} for t, m in msgs.items()]

        def label(t, model=model):
            res, usage = call([{"role": "user", "content": PROMPT.format(levels=LEVELS_V2, text=t["key"])}],
                              LABEL_SCHEMA, k, max_tokens=20, temperature=0, model=model)
            return {**t, **res}, usage
        print(f"labeler {tag}: {model}", file=sys.stderr)
        run(tasks, label, os.path.join(DATA, f"v2_labels_{tag}.jsonl"), workers=32)


if __name__ == "__main__":
    main()
