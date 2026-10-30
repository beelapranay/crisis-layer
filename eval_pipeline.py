"""Tunes the classifier thresholds on validation data, then scores rules alone, the classifier alone and the combined
pipeline once on the test sets (tests.py: dev, heldout, idioms).

Threshold choice, on the real-message validation split (val_real.jsonl) only, one per level, from the classifier's
scores on messages labelled "none":
- T_RISK: the score that MAX_FALSE_ALARM of them reach (they would get a check-in)
- T_CRISIS: the P(crisis) that MAX_FALSE_CRISIS of them reach (they would get the crisis screen)
A recall target instead (the first versions) is unstable: recall can fall off a cliff between grid steps and drag
the false-alarm rate to 30-65%.

    ../neuron-explorer/.venv/bin/python eval_pipeline.py
"""
import json
import os
import time

from pipeline import Classifier, clf_level, combine
from rules import check_input
from tests import BORDERLINE, RISK_CATS, sets

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = os.environ.get("CRISIS_TAG", "")  # report suffix, to compare classifiers
MAX_FALSE_ALARM, MAX_FALSE_CRISIS = 0.10, 0.02


def val_score(rows, probs, t_risk, t_crisis):
    lv = [combine(r["text"], p, t_risk, t_crisis)[0] for r, p in zip(rows, probs)]
    risk = [l != "none" for l in lv]
    rec = sum(x for x, r in zip(risk, rows) if r["label"] != "none") / sum(r["label"] != "none" for r in rows)
    crec = sum(l == "crisis" for l, r in zip(lv, rows) if r["label"] == "crisis") / sum(r["label"] == "crisis" for r in rows)
    fa = sum(x for x, r in zip(risk, rows) if r["label"] == "none") / sum(r["label"] == "none" for r in rows)
    return rec, crec, fa


def test_score(name, items, levels):
    risk = [(t, l) for (t, c), l in zip(items, levels) if c in RISK_CATS]
    calm = [(t, l) for (t, c), l in zip(items, levels) if c not in RISK_CATS and t not in BORDERLINE]
    out = {"n": len(items)}
    if risk:
        out["risk_flagged"] = f"{sum(l != 'none' for _, l in risk)}/{len(risk)}"
        out["risk_crisis_screen"] = f"{sum(l == 'crisis' for _, l in risk)}/{len(risk)}"
        out["missed"] = [t for t, l in risk if l == "none"]
    out["false_alarms"] = f"{sum(l != 'none' for _, l in calm)}/{len(calm)}"
    out["false_crisis"] = f"{sum(l == 'crisis' for _, l in calm)}/{len(calm)}"
    out["alarms"] = [f"[{l}] {t}" for t, l in calm if l != "none"]
    return out


def main():
    clf = Classifier()
    val = [json.loads(s) for s in open(os.path.join(HERE, "data", "val_real.jsonl"))]  # real messages only
    vp = clf.probs([r["text"] for r in val])
    calm = [p for r, p in zip(val, vp) if r["label"] == "none"]
    risk_scores = sorted(p[1] + p[2] for p in calm)
    crisis_scores = sorted(p[2] for p in calm)
    tr = round(risk_scores[int((1 - MAX_FALSE_ALARM) * len(calm))], 3)
    tc = round(crisis_scores[int((1 - MAX_FALSE_CRISIS) * len(calm))], 3)
    rec, crec, fa = val_score(val, vp, tr, tc)
    print(f"thresholds from validation: T_RISK={tr} T_CRISIS={tc} -> val risk recall {rec:.3f}, "
          f"crisis recall {crec:.3f}, false alarms {fa:.3f}")

    report = {"thresholds": {"T_RISK": tr, "T_CRISIS": tc}, "val": {"risk_recall": rec, "crisis_recall": crec,
                                                                     "false_alarm": fa}}
    for name, items in sets().items():
        texts = [t for t, _ in items]
        probs = clf.probs(texts)
        systems = {"rules": [check_input(t)[0] for t in texts],
                   "classifier": [clf_level(p, tr, tc) for p in probs],
                   "combined": [combine(t, p, tr, tc)[0] for t, p in zip(texts, probs)]}
        report[name] = {}
        print(f"\n== {name} ({len(items)} messages)")
        for sname, levels in systems.items():
            s = test_score(name, items, levels)
            report[name][sname] = s
            line = "  ".join(f"{k} {v}" for k, v in s.items() if k not in ("missed", "alarms", "n"))
            print(f"   {sname:10} {line}")
        comb = report[name]["combined"]
        for t in comb.get("missed", []):
            print(f"      MISS  {t}")
        for a in comb["alarms"]:
            print(f"      ALARM {a}")

    test_path = os.path.join(HERE, "data", "test_real.jsonl")  # frozen real split, labelled under v2
    if os.path.exists(test_path):
        test = [json.loads(x) for x in open(test_path)]
        rec, crec, fa = val_score(test, clf.probs([r["text"] for r in test]), tr, tc)
        calm = [r for r in test if r["label"] == "none"]
        fc = sum(combine(r["text"], p, tr, tc)[0] == "crisis" for r, p in zip(calm, clf.probs([r["text"] for r in calm])))
        report["test_real"] = {"n": len(test), "risk_recall": rec, "crisis_recall": crec, "false_alarm": fa,
                               "false_crisis": fc / len(calm)}
        print(f"\n== test_real ({len(test)} real messages, frozen): risk recall {rec:.3f}, crisis recall {crec:.3f}, "
              f"false alarms {fa:.3f}, false crisis screens {fc}/{len(calm)}")

    t0 = time.time()
    for t, _ in sets()["heldout"]:
        clf.probs([t])
    print(f"\nclassifier latency, one message at a time on {clf.dev}: {1000 * (time.time() - t0) / 80:.1f} ms")
    json.dump(report, open(os.path.join(HERE, f"pipeline_report{TAG}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
