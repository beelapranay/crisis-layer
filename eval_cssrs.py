"""Scores the crisis layer on the Reddit C-SSRS Suicide Dataset (Gaur et al. 2019, CC BY 4.0,
https://zenodo.org/records/2667859): 500 Reddit users, each labelled by psychiatrists on a C-SSRS-based scale as
Supportive, Indicator, Ideation, Behavior or Attempt. Test data only; never used for training or thresholds.

Each post is rated on its own and a user gets their highest post level. Expected levels: Supportive -> none
(people helping others, so a flag is a false alarm), Indicator and Ideation -> at least checkin, Behavior and
Attempt -> crisis.

    ../neuron-explorer/.venv/bin/python eval_cssrs.py    (expects external/cssrs/500_Reddit_users_posts_labels.csv)
"""
import ast
import collections
import csv
import json
import os

from pipeline import T_CRISIS, T_RISK, Classifier, clf_level, combine
from rules import check_input

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = os.environ.get("CRISIS_TAG", "")  # report suffix, to compare classifiers
PATH = os.path.join(HERE, "external", "cssrs", "500_Reddit_users_posts_labels.csv")
RANK = {"none": 0, "checkin": 1, "crisis": 2}
WANT = {"Supportive": 0, "Indicator": 1, "Ideation": 1, "Behavior": 2, "Attempt": 2}


def users():
    csv.field_size_limit(10 ** 8)
    out = []
    for r in csv.DictReader(open(PATH, encoding="utf-8", errors="replace")):
        try:
            posts = ast.literal_eval(r["Post"])
        except (ValueError, SyntaxError):
            posts = [r["Post"]]
        out.append((r["User"], r["Label"], [p for p in posts if p.strip()]))
    return out


def main(t_risk=None, t_crisis=None):
    tuned = os.path.join(HERE, f"pipeline_report{TAG}.json")  # thresholds eval_pipeline.py chose on validation
    if t_risk is None and os.path.exists(tuned):
        t = json.load(open(tuned))["thresholds"]
        t_risk, t_crisis = t["T_RISK"], t["T_CRISIS"]
    t_risk = T_RISK if t_risk is None else t_risk
    t_crisis = T_CRISIS if t_crisis is None else t_crisis
    clf = Classifier()
    U = users()
    posts = [p for _, _, ps in U for p in ps]
    probs = iter(clf.probs(posts))
    table = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for _, label, ps in U:
        best = {"rules": 0, "classifier": 0, "combined": 0}
        for p in ps:
            pr = next(probs)
            best["rules"] = max(best["rules"], RANK[check_input(p)[0]])
            best["classifier"] = max(best["classifier"], RANK[clf_level(pr, t_risk, t_crisis)])
            best["combined"] = max(best["combined"], RANK[combine(p, pr, t_risk, t_crisis)[0]])
        for s, v in best.items():
            table[s][label][v] += 1
    print(f"{len(U)} users, {len(posts)} posts; thresholds T_RISK={t_risk} T_CRISIS={t_crisis}")
    report = {}
    for s, by in table.items():
        n_risk = sum(sum(c.values()) for l, c in by.items() if WANT[l] >= 1)
        flagged = sum(c[1] + c[2] for l, c in by.items() if WANT[l] >= 1)
        n_high = sum(sum(c.values()) for l, c in by.items() if WANT[l] == 2)
        high_crisis = sum(c[2] for l, c in by.items() if WANT[l] == 2)
        sup = by["Supportive"]
        report[s] = {"risk_flagged": flagged / n_risk, "behavior_attempt_crisis": high_crisis / n_high,
                     "supportive_flagged": (sup[1] + sup[2]) / sum(sup.values()),
                     "by_label": {l: dict(c) for l, c in by.items()}}
        print(f"\n== {s}: risk users flagged {flagged}/{n_risk} ({100 * flagged / n_risk:.0f}%), behavior/attempt "
              f"to crisis {high_crisis}/{n_high} ({100 * high_crisis / n_high:.0f}%), supportive flagged "
              f"{sup[1] + sup[2]}/{sum(sup.values())}")
        for l in WANT:
            c = by[l]
            print(f"   {l:10} none {c[0]:3}  checkin {c[1]:3}  crisis {c[2]:3}")
    json.dump(report, open(os.path.join(HERE, f"cssrs_report{TAG}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
