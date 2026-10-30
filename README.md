# Crisis layer (prototype)

Safety checks that run beside the chat model in the mental wellbeing app, so that crisis handling does not depend on
LFM3-1.7B noticing risk (with the best system prompt it still missed ~13% of held-out crisis messages).

- `rules.py`: tier 1, regex only, stdlib only (ports to Swift's NSRegularExpression).
  - `check_input(text)` → `"none" | "checkin" | "crisis"` plus the rules that fired.
  - `strip_resources(reply)` drops every sentence naming a crisis resource; the app shows its own verified card.
  - `check_output(reply, level)` blocks unknown phone numbers, doses, very-low-calorie plans, validation of
    self-harm, concealment tips, and goodbye letters for at-risk users.
- `eval_rules.py`: scores the rules on the supporter eval prompts and the ~18.6k judged replies in
  `neuron-explorer/stats/lfm3-1.7b-persona/eval/`. Held-out failures stay hidden unless `--show-heldout`.
- Tier 2, a fine-tuned RoBERTa-base classifier (none / checkin / crisis), combined with the rules in `pipeline.py`:
  - data: `gen_data.py` (synthetic messages), `fetch_real.py` (real Reddit text, see Data), `relabel.py` (v2
    labels from two LLMs; messages kept where both agree), `build_v2.py` (splits, a frozen real test split)
  - `train.py` (any Hugging Face encoder via `CRISIS_BASE`), `eval_pipeline.py` (thresholds on real validation,
    then every test set), `eval_cssrs.py` (psychiatrist-labelled C-SSRS users), `tests.py` (test sets)
  - not in git: `data/`, `external/` and `model*/` (rebuild with the scripts; the shipped model is
    `model_roberta_v2/`, ~500 MB fp32). The shipped PyTorch and Core ML models are backed up in the private
    Hugging Face repo `pranaybeela/crisis-classifier` (`pytorch/`, `coreml/`)

Pipeline in the app: a crisis-rated message shows the fixed crisis screen instead of a model reply; a check-in adds a
gentle safety question to the model's reply; every reply is resource-stripped and passed through `check_output`.

## App behaviour per level

The classifier's check-in threshold is set low on purpose, so roughly a third to a half of plainly sad messages
trigger a check-in. That only works if a check-in costs the user almost nothing:

- **none**: normal chat. No crisis resources anywhere in the reply (they are stripped) or the UI.
- **check-in**: the model's reply is shown as usual, followed by one small, dismissible line under it, for example
  "How are you holding up right now? If you're ever not feeling safe, support is one tap away." with two quiet
  buttons: "I'm okay" and "I'm not safe". No modal, no hotline numbers in the line itself.
  - At most one check-in every 10 messages or 30 minutes, unless the level rises or the message is a new kind of
    risk (the rules fired where they had not before).
  - "I'm okay" lowers sensitivity for the rest of the session only for the same kind of message; it never turns
    the check-in off.
  - "I'm not safe" opens the crisis screen.
- **crisis**: the crisis screen replaces the model's reply: verified resources for the user's region (988 call and
  text in the US), emergency services, a short grounding exercise, and a way back to the chat. The wording is fixed
  and reviewed, never generated. The session stays at heightened sensitivity afterwards.

The crisis screen needs either a classifier P(crisis) above a strict threshold or a high-precision rule ("kill
myself", "suicide", "overdose", "already took", ...). Other rule hits ("this traffic is killing me", "cut myself
some slack") only ever produce a check-in.

## Tier 2 results (2026-10-29)

Shipped: RoBERTa-base trained on 18,963 messages (~70% real) with v2 labels, thresholds `T_RISK = 0.267` (check-in
for 10% of safe validation messages) and `T_CRISIS = 0.138` (crisis screen for 2%), rules + classifier as in
`pipeline.py`.

| | rules only | rules + RoBERTa v1 labels | **rules + RoBERTa v2 labels** |
|---|---|---|---|
| risk test prompts flagged (dev + held-out) | 105/105 | 105/105 | **105/105** |
| everyday test prompts flagged | 1/163 | 61/163 | **36/163** |
| everyday test prompts sent to the crisis screen | 0 | 10 | **4** |
| idioms flagged / sent to the crisis screen | 12 / 10 | 13 / 4 | 18 / 3 |
| C-SSRS at-risk users flagged (392) | 76% | 97% | 95% |
| C-SSRS behavior/attempt users sent to the crisis screen (122) | 89% | 98% | 97% |
| frozen real test split (944): risk flagged / crisis screened / false alarms / false crisis | — | — | 96% / 96% / 11% / 1.8% |

Classifier latency on a GPU: ~5–6 ms per message.

What changed the most:
- **Real text.** Synthetic training data alone (MiniLM) caught 93% of C-SSRS at-risk users but flagged a third of
  everyday prompts; adding ~14k real Reddit and GoEmotions messages (labelled by two LLMs) lifted that to 97–99%.
- **Label definitions (v2).** v1 counted "hopelessness" as a check-in sign and told the labeller to pick the higher
  level when unsure, so it labelled ~19% of plainly sad messages as check-in and the classifier learned "sad =
  risk". v2 makes sadness, numbness, guilt and other depression symptoms "none" unless they come with a sign of
  danger; everyday prompts flagged fell from 61 to 36.
- **Model size helped a little.** At equal false-alarm rates RoBERTa-base (125M) beat MiniLM-L6 (22M) on every
  measure, most on C-SSRS (AUROC 0.73 → 0.78) and idioms.

Caveats:
- All training and validation labels come from LLMs, not clinicians, and the frozen test split shares that labelling
  policy; only C-SSRS has expert labels. The v2 policy rates C-SSRS "Indicator" users (risk factors without talk of
  death) as safe more often, which a clinician should review.
- The hand-written test sets (dev, held-out, idioms) have been looked at four times while developing; treat them as
  development numbers. The frozen real split was scored once for v2 (v1 had trained on those messages).
- C-SSRS rates a user by their worst post, and "Supportive" users discuss suicide while helping others, so its
  supportive-flagged numbers overstate false alarms.

## On the iPhone (Core ML)

`export_coreml.py` (own venv: coremltools 9.0, torch 2.7.0, transformers 5.20.1) writes `coreml/` (not in git):

- `CrisisClassifier.mlpackage`, 250 MB, fp16, iOS 18+: a multifunction model with one function per input length
  (`len16`, `len32`, `len64`, `len128`, `len256`) sharing one copy of the weights. A single model with flexible
  (enumerated) lengths segfaulted on the CPU and Neural Engine for 128+ tokens; fixed-length functions do not.
- `tokenizer.json` and `crisis_config.json` (thresholds, function names, pad token, the combining rule).

Checked against PyTorch on 1,252 test messages through the Neural Engine path: mean probability difference 0.0007,
largest 0.022; 6 messages changed level, all within 0.003 of a threshold. About 3 ms per message on an M3 Mac's
Neural Engine (CPU_AND_NE); expect the same order on an iPhone 16 Pro.

In the app: tokenize with swift-transformers from `tokenizer.json`; pad to the smallest length that fits and load
that function (`MLModelConfiguration.functionName`, compute units `.cpuAndNeuralEngine`); for messages over 256
tokens, score 256-token windows and take the highest risk; then apply the rules and thresholds from
`crisis_config.json` exactly as `pipeline.combine` does.

## Data

- Reddit C-SSRS Suicide Dataset, Gaur et al. 2019, CC BY 4.0 (https://zenodo.org/records/2667859): test only.
- Reddit Mental Health Dataset, Low et al. 2020, ODC PDDL (https://zenodo.org/records/3941387): sampled 2019 posts
  and post openings for training; re-use is subject to Reddit's API terms.
- GoEmotions, Google, Apache 2.0 (https://huggingface.co/datasets/google-research-datasets/go_emotions): short
  comments for training.
- Training on Reddit text is acceptable here because the project is research / non-commercial; revisit Reddit's
  data terms before any commercial use.

## Tier 1 results (2026-10-29)

| | dev (200) | held-out (80) |
|---|---|---|
| risk messages flagged (≥ check-in) | 60/60 | 44/45 before one post-hoc fix |
| everyday/venting false alarms | 1/132 | 0/31 |
| judge-flagged replies a user would still see (unsteered app/v5 replies) | 1 harmful, 6 missed-crisis (on check-in messages) | 1 harmful, 5 missed-crisis |

Caveats:
- The same person wrote the test prompts and the rules, so recall is optimistic; real users phrase things in ways
  neither anticipated.
- Idioms: 12/28 everyday idioms were flagged, 10 at crisis level ("this traffic is killing me", "cut myself some
  slack", "end things with my boyfriend", "a rope for rock climbing"). Rules alone are too blunt to trigger the full
  crisis screen; tier 2 (a small classifier) has to confirm or downgrade.
- `check_output` alone catches under half of the judge-flagged replies; it works because crisis-rated messages never
  reach the model. The remaining leaks are diet and drinking advice on check-in messages.

## Safety and license

This is a research prototype, not a medical device or a substitute for professional care, and it is not validated
for use with real people in crisis. Crisis resources in the app must be verified, region-appropriate and maintained by
people; clinician review is needed before any deployment.

The test prompts in `../neuron-explorer/sweep/supporter_prompts.py` describe suicide, self-harm and abuse on purpose.

MIT licensed; see `LICENSE`.
