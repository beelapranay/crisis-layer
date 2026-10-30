"""Exports the tier 2 classifier to Core ML for the iPhone app, then checks it against the PyTorch model.

Writes coreml/:
- CrisisClassifier.mlpackage: a multifunction ML program (iOS 18+), fp16, one function per input length L in
  SHAPES ("len16" ... "len256"), sharing one copy of the weights. Inputs input_ids and attention_mask (int32,
  [1, L]), output probs [1, 3] = P(none, checkin, crisis). The app pads each message to the smallest L that fits
  and loads that function (MLModelConfiguration.functionName). One model with EnumeratedShapes crashed (segfault)
  on the CPU and Neural Engine for L >= 128 with coremltools 9.0; fixed-length functions do not.
- tokenizer.json: the model's tokenizer (load it with swift-transformers).
- crisis_config.json: labels, thresholds, function names and the pad token, so the app applies the same rules.

Verification: every message in the test sets (tests.py) and the frozen real test split, through both models on the
Neural Engine path; reports the largest probability difference and how often the level differs. Then times
single-message predictions on this Mac (a rough guide for the iPhone, whose Neural Engine is similar).

    .venv/bin/python export_coreml.py [--verify-only]      (separate venv: coremltools 9.0, torch 2.7.0)
"""
import json
import os
import shutil
import sys
import tempfile
import time

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pipeline import T_CRISIS, T_RISK, clf_level
from tests import sets

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "model_roberta_v2")
OUT = os.path.join(HERE, "coreml")
PATH = os.path.join(OUT, "CrisisClassifier.mlpackage")
SHAPES = (16, 32, 64, 128, 256)
LABELS = ["none", "checkin", "crisis"]


class Probs(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        # a ready-made 4D additive mask makes transformers skip its own mask building, whose ops (new_ones, ...)
        # coremltools cannot convert; -1e4 rather than the dtype minimum so it stays finite in fp16
        mask = (1.0 - attention_mask.float())[:, None, None, :] * -1e4
        logits = self.model(input_ids=input_ids.long(), attention_mask=mask).logits
        return torch.softmax(logits.float(), -1)


def fn(L):
    return f"len{L}"


def bucket(n):
    return next(s for s in SHAPES if s >= n)


def encode(tok, text):
    ids = tok(text, truncation=True, max_length=SHAPES[-1])["input_ids"]
    L = bucket(len(ids))
    pad = [tok.pad_token_id] * (L - len(ids))
    return L, np.array([ids + pad], dtype=np.int32), np.array([[1] * len(ids) + [0] * len(pad)], dtype=np.int32)


def export(tok, model):
    wrapped = Probs(model).eval()
    desc = None
    with tempfile.TemporaryDirectory() as tmp:
        for L in SHAPES:
            ex = (torch.ones(1, L, dtype=torch.int32), torch.ones(1, L, dtype=torch.int32))
            with torch.no_grad():
                traced = torch.jit.trace(wrapped, ex)
            ml = ct.convert(traced, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
                            compute_precision=ct.precision.FLOAT16,
                            inputs=[ct.TensorType(name="input_ids", shape=(1, L), dtype=np.int32),
                                    ct.TensorType(name="attention_mask", shape=(1, L), dtype=np.int32)],
                            outputs=[ct.TensorType(name="probs")])
            p = os.path.join(tmp, f"{fn(L)}.mlpackage")
            ml.save(p)
            if desc is None:
                desc = ct.utils.MultiFunctionDescriptor()
            desc.add_function(p, src_function_name="main", target_function_name=fn(L))
            print(f"converted L={L}", flush=True)
        desc.default_function_name = fn(64)
        if os.path.exists(PATH):
            shutil.rmtree(PATH)
        ct.utils.save_multifunction(desc, PATH)
    shutil.copy(os.path.join(SRC, "tokenizer.json"), OUT)
    json.dump({"labels": LABELS, "T_RISK": T_RISK, "T_CRISIS": T_CRISIS, "pad_token_id": tok.pad_token_id,
               "max_length": SHAPES[-1], "functions": {str(L): fn(L) for L in SHAPES},
               "rule": "crisis if P(crisis) >= T_CRISIS or a high-precision rule; checkin if "
                       "P(checkin)+P(crisis) >= T_RISK or any rule; else none"},
              open(os.path.join(OUT, "crisis_config.json"), "w"), indent=1)


def load(units):
    return {L: ct.models.MLModel(PATH, compute_units=units, function_name=fn(L)) for L in SHAPES}


def main():
    os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(SRC)
    model = AutoModelForSequenceClassification.from_pretrained(SRC, attn_implementation="eager").eval()
    if "--verify-only" not in sys.argv or not os.path.exists(PATH):
        export(tok, model)
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(PATH) for f in fs)
    print(f"saved {PATH} ({size / 1e6:.0f} MB)", flush=True)

    texts = [t for s in sets().values() for t, _ in s]
    texts += [json.loads(x)["text"] for x in open(os.path.join(HERE, "data", "test_real.jsonl"))]
    for units in (ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.ALL):
        ml = load(units)
        diffs, flips = [], 0
        for t in texts:
            L, ids, mask = encode(tok, t)
            with torch.no_grad():
                ref = Probs(model)(torch.from_numpy(ids), torch.from_numpy(mask))[0].numpy()
            got = ml[L].predict({"input_ids": ids, "attention_mask": mask})["probs"][0]
            diffs.append(float(np.abs(ref - got).max()))
            flips += clf_level(ref.tolist(), T_RISK, T_CRISIS) != clf_level(got.tolist(), T_RISK, T_CRISIS)
        d = np.array(diffs)
        print(f"{units.name:10} {len(texts)} messages: max |dP| {d.max():.4f}, mean {d.mean():.5f}, "
              f"99th pct {np.percentile(d, 99):.4f}; classifier level differs on {flips}", flush=True)

    for units in (ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.ALL, ct.ComputeUnit.CPU_ONLY):
        ml = load(units)
        for L in (32, 128):
            _, ids, mask = encode(tok, "I've been feeling really down lately and I don't know who to talk to.")
            x = {"input_ids": np.pad(ids, ((0, 0), (0, L - ids.shape[1])), constant_values=tok.pad_token_id),
                 "attention_mask": np.pad(mask, ((0, 0), (0, L - mask.shape[1])))}
            ml[L].predict(x)
            t0 = time.time()
            for _ in range(20):
                ml[L].predict(x)
            print(f"   {units.name:10} L={L:3}: {1000 * (time.time() - t0) / 20:.1f} ms per message (this Mac)",
                  flush=True)


if __name__ == "__main__":
    main()
