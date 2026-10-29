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

Pipeline in the app: a crisis-rated message shows the fixed crisis screen instead of a model reply; a check-in adds a
gentle safety question to the model's reply; every reply is resource-stripped and passed through `check_output`.

## Results (2026-10-29)

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
