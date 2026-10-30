"""Labels fetch_real.py's real messages for classifier training with two different LLMs on OpenRouter; build_data.py
keeps the messages both put at the same level. Resumable. --short labels real_short.jsonl.

    python3 label_real.py [--short]
"""
import json
import os
import sys

from gen_data import LABEL_SCHEMA, LEVELS, call, key, run

HERE = os.path.dirname(os.path.abspath(__file__))
SHORT = "--short" in sys.argv
SRC = os.path.join(HERE, "external", "real_short.jsonl" if SHORT else "real_sample.jsonl")
LABELERS = {"a": "deepseek/deepseek-v4.1-flash-1017", "b": "deepseek/deepseek-v4-flash"}
PROMPT = ("Classify this message sent to a mental wellbeing chatbot into one level, judging the risk to the person "
          "writing it (or to someone they might harm). Someone describing another person's crisis, giving support, "
          "or talking about the past while now safe is not at risk themselves.\n\n{levels}\n\nWhen unsure between two "
          "levels, pick the higher one.\n\nMessage:\n<<<{text}>>>")


def main():
    k = key()
    rows = [json.loads(s) for s in open(SRC)]
    for tag, model in LABELERS.items():
        tasks = [{"key": r["text"], "source": r["source"], "subreddit": r.get("subreddit"), "labeler": model}
                 for r in rows]

        def label(t, model=model):
            res, usage = call([{"role": "user", "content": PROMPT.format(levels=LEVELS, text=t["key"])}],
                              LABEL_SCHEMA, k, max_tokens=20, temperature=0, model=model)
            return {**t, **res}, usage
        print(f"labeler {tag}: {model}", file=sys.stderr)
        run(tasks, label, os.path.join(HERE, "data", f"real{'_short' if SHORT else ''}_labels_{tag}.jsonl"), workers=24)


if __name__ == "__main__":
    main()
