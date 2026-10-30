"""The crisis layer's input check: regex rules (rules.py) combined with the tier 2 classifier (train.py).

Each level has its own trigger, because the two mistakes cost different amounts: a wrong check-in costs a gentle
question (the app keeps it small, see README), a wrong crisis screen costs trust.
- crisis screen: P(crisis) >= T_CRISIS (a high threshold) or a high-precision rule (HIGH_PRECISION)
- check-in: P(checkin) + P(crisis) >= T_RISK (a low threshold), or any rule. Rules alone fire on idioms ("this
  traffic is killing me"), so outside HIGH_PRECISION they never reach the crisis screen.
"""
import functools
import os
import re

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rules import LEVELS, _norm, check_input

HERE = os.path.dirname(os.path.abspath(__file__))
# RoBERTa v2: check-in at 10% false alarms, crisis screen at 2% false crisis screens on real-text validation
T_RISK, T_CRISIS = 0.267, 0.138

HIGH_PRECISION = [re.compile(p) for p in (
    r"\bkill(ing)? my ?self\b", r"\bsuicid", r"\b(end|ending|take|taking) my (own )?life\b",
    r"\bwant(ed)? to die\b", r"\boverdos", r"\balready took\b", r"\btonight's the night\b",
    r"\b(took|swallowed) (a lot of|all|too many|a bunch of|the whole) (my )?(pills|meds|tablets)\b",
    r"\blethal dose\b", r"\b(cut|cutting|burn|burned|burning) my ?self\b(?! (some )?slack)(?! out)",
)]


class Classifier:
    def __init__(self, path=os.environ.get("CRISIS_MODEL", os.path.join(HERE, "model_roberta_v2")), dev=None):
        self.dev = dev or ("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.dev).eval()

    def probs(self, texts, bs=64):
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), bs):
                x = self.tok(texts[i:i + bs], padding=True, truncation=True, max_length=256,
                             return_tensors="pt").to(self.dev)
                out.append(torch.softmax(self.model(**x).logits.float(), -1).cpu())
        return torch.cat(out).tolist()


def clf_level(p, t_risk=None, t_crisis=None):
    t_risk = T_RISK if t_risk is None else t_risk
    t_crisis = T_CRISIS if t_crisis is None else t_crisis
    if p[2] >= t_crisis:
        return "crisis"
    return "checkin" if p[1] + p[2] >= t_risk else "none"


def combine(text, p, t_risk=None, t_crisis=None):
    """Returns (level, why) for one message given its classifier probabilities p = [none, checkin, crisis]."""
    c = clf_level(p, t_risk, t_crisis)
    r, hits = check_input(text)
    hp = high_precision(text)
    if c == "crisis" or hp:
        return "crisis", {"classifier": c, "rules": hits, "high_precision": hp}
    if r != "none" or c == "checkin":
        return "checkin", {"classifier": c, "rules": hits, "high_precision": hp}
    return "none", {"classifier": c, "rules": hits, "high_precision": hp}


@functools.lru_cache(maxsize=1 << 16)
def high_precision(text):
    t = _norm(text)
    return any(rx.search(t) for rx in HIGH_PRECISION)


def rate(texts, clf, **kw):
    return [combine(t, p, **kw) for t, p in zip(texts, clf.probs(texts))]


assert LEVELS == ("none", "checkin", "crisis")
