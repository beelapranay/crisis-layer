"""Generates labelled training messages for the tier 2 crisis classifier with an LLM on OpenRouter.

Two passes, both resumable:
1. generate: for each subcategory x writing style, ask for a batch of messages a user might send a support chatbot.
   Messages come from category descriptions only; the model never sees the dev, held-out or idiom test prompts.
2. label: every message is labelled again in a separate call that sees only the message and the level definitions.
   build_data.py keeps messages where both labels agree.

    python3 gen_data.py generate
    python3 gen_data.py label

The key is read from ~/.openrouter_key (or OPENROUTER_API_KEY) and never written anywhere.
"""
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-v4.1-flash-1017"

LEVELS = """- none: no sign of risk to anyone's life or safety. Everyday stress, sadness, loneliness, grief, venting,
  ordinary health, diet or fitness questions, figures of speech ("this traffic is killing me"), talk about fiction,
  news or the past when the person is now safe.
- checkin: possible but unclear risk, worth a gentle question about safety: passive or indirect wishes to die or
  disappear, hopelessness, feeling like a burden, preoccupation with death, risky drinking or drug use to cope,
  disordered-eating thoughts, stopping psychiatric medication on their own, wanting to punish themselves.
- crisis: clear risk now: suicidal intent, a plan, access to means, a recent or ongoing attempt or overdose, active
  self-harm, being abused or in danger, voices telling them to harm themselves, intent to harm someone else, acute
  eating-disorder danger, or asking for methods, lethal amounts or ways to hide self-harm or not eating."""

# (expected level, name, description); the expected level steers generation, the label pass decides.
SUBCATS = [
    ("crisis", "explicit_intent", "states they want to kill themselves or end their life, directly"),
    ("crisis", "plan_or_time", "has a plan, a date or a time for ending their life"),
    ("crisis", "means_access", "has gathered or has access to means (pills, a weapon, a rope, a high place)"),
    ("crisis", "attempt_in_progress", "has just taken something, hurt themselves badly, or is in the middle of an attempt"),
    ("crisis", "active_self_harm", "is currently cutting, burning or otherwise injuring themselves"),
    ("crisis", "goodbye_behaviour", "is saying goodbye, writing final letters, giving possessions away, settling affairs"),
    ("crisis", "abuse_danger", "is being hurt, threatened, controlled or assaulted by someone, or is unsafe at home"),
    ("crisis", "psychosis_command", "hears voices or has beliefs telling them to hurt themselves"),
    ("crisis", "harm_to_others", "has urges or plans to hurt someone else, including their own child"),
    ("crisis", "ed_acute", "describes acute eating-disorder danger: days without eating, fainting, purging after every meal"),
    ("crisis", "method_request", "asks for methods, lethal doses or amounts, or the most painless or effective way to die or self-harm"),
    ("crisis", "concealment_request", "asks how to hide self-harm, injuries, not eating, or risk from family or clinicians"),
    ("crisis", "goodbye_subtle", "is quietly putting their affairs in order (paying bills, cancelling plans, giving away "
     "things, saying thanks) in a way that implies they won't be around, without saying suicide"),
    ("crisis", "self_harm_euphemism", "refers to self-harm or an attempt only indirectly ('did it again', 'relapsed', "
     "'did something to myself', 'it's deeper this time')"),
    ("crisis", "harm_to_others_subtle", "hints at a plan or weapon to hurt a specific person, or urges to hurt a child, "
     "without stating it plainly"),
    ("checkin", "passive_ideation", "wishes they would not wake up or could disappear, without a plan"),
    ("checkin", "hopelessness", "says there is no point, no way out, things will never get better"),
    ("checkin", "burden", "feels like a burden or that others would be better off without them"),
    ("checkin", "death_preoccupation", "keeps thinking about death or their own funeral, vaguely"),
    ("checkin", "substance_coping", "drinks or uses drugs heavily to cope or numb out"),
    ("checkin", "ed_thoughts", "has disordered eating thoughts, wants to restrict or punish themselves for eating"),
    ("checkin", "stopping_meds", "wants to stop psychiatric medication on their own"),
    ("checkin", "self_punishment", "wants to punish themselves or believes they deserve to suffer"),
    ("none", "everyday_stress", "is stressed about work, school, money or deadlines"),
    ("none", "sadness_loneliness", "feels sad, lonely, left out or unmotivated, without any sign of risk"),
    ("none", "relationships", "has conflict or heartbreak with a partner, friend or family member"),
    ("none", "venting", "is venting frustration or anger about their day"),
    ("none", "anxiety", "has anxiety, worry, panic or trouble sleeping"),
    ("none", "grief", "is grieving a person or pet who died, with no risk to themselves"),
    ("none", "hn_idioms", "uses violent or death figures of speech harmlessly ('killing me', 'dying to', 'dead tired', 'kill it', 'could die laughing')"),
    ("none", "hn_benign_verbs", "uses words like cut, burn, hurt, end, overdose or disappear harmlessly (cutting carbs, burned out, hurt my knee, ending a lease, overdosing on coffee)"),
    ("none", "hn_benign_objects", "mentions ropes, bridges, heights, pills or medication in harmless contexts (climbing, photography, a cold, disposing of old meds)"),
    ("none", "hn_medical", "asks ordinary medication or dosage questions for a normal health problem"),
    ("none", "hn_diet_fitness", "asks about healthy dieting, calorie counting, fasting for religion or intermittent fasting, gym goals"),
    ("none", "hn_fiction_media", "talks about death or suicide in a book, show, game, news story, essay or psychology class"),
    ("none", "hn_life_admin", "mentions goodbye parties, writing a will after buying a house, letters to a future self, giving away clothes"),
    ("none", "hn_recovered", "talks about past struggles they have recovered from and feeling safe and proud now"),
    ("none", "hn_dark_humor", "makes hyperbolic or gallows-humour complaints with no real intent"),
]
STYLES = [
    "short texting style: lowercase, abbreviations, occasional typos",
    "one or two plain sentences",
    "a longer, rambling message with context about their life",
    "a teenager writing",
    "an older adult writing formally",
    "a non-native English speaker",
    "with emoji or internet slang",
    "a reply in the middle of a conversation, without greeting",
]
# Real messages are often very short; the first models over-reacted to short everyday messages (test median 8 words
# against 18-21 in training). Two batches per subcategory in this style.
SHORT = ["very short: 3 to 10 words, plain", "very short: 3 to 10 words, casual texting"]


def key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k:
        with open(os.path.expanduser("~/.openrouter_key")) as f:
            k = f.read().strip()
    return k


def call(messages, schema, k, max_tokens, temperature, retries=5, model=MODEL):
    body = json.dumps({"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
                       "reasoning": {"enabled": False},
                       "response_format": {"type": "json_schema", "json_schema": schema}}).encode()
    err = None
    for attempt in range(retries):
        req = urllib.request.Request(URL, data=body, headers={"Authorization": f"Bearer {k}",
                                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.load(r)
            return json.loads(resp["choices"][0]["message"]["content"]), resp.get("usage", {})
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read()[:200]!r}"
            if e.code not in (408, 429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
            err = f"{type(e).__name__}: {e}"
        time.sleep(2 ** attempt + random.random())
    return {"error": err}, {}


GEN_SCHEMA = {"name": "batch", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["messages"],
    "properties": {"messages": {"type": "array", "items": {"type": "string"}}}}}
LABEL_SCHEMA = {"name": "label", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["level"],
    "properties": {"level": {"type": "string", "enum": ["none", "checkin", "crisis"]}}}}


def gen_prompt(level, desc, style, n):
    return (f"You are helping build a safety classifier for a mental wellbeing support app. It must recognise risk "
            f"accurately, including subtle and indirect wording, and must not over-react to harmless messages.\n\n"
            f"Write {n} different messages a user might send to the app's chatbot. Each message comes from someone who "
            f"{desc}. Writing style: {style}. Vary the person, situation, wording and length; do not reuse phrases "
            f"across messages; do not number them. Write only the user's message, no labels or commentary.\n\n"
            f"For reference, the classifier's levels are:\n{LEVELS}\nThese messages are meant to be level '{level}'.")


def run(tasks, fn, out, workers):
    done = set()
    if os.path.exists(out):
        done = {json.loads(s)["key"] for s in open(out)}
    todo = [t for t in tasks if t["key"] not in done]
    print(f"{len(todo)} to do ({len(done)} already done)", file=sys.stderr)
    tin = tout = errs = 0
    with open(out, "a") as f, ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fn, t) for t in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            row, usage = fut.result()
            errs += "error" in row
            tin, tout = tin + usage.get("prompt_tokens", 0), tout + usage.get("completion_tokens", 0)
            f.write(json.dumps(row) + "\n")
            f.flush()
            if n % 200 == 0 or n == len(todo):
                print(f"{n}/{len(todo)}, {errs} errors, {tin} in / {tout} out tokens", file=sys.stderr)


def main():
    os.makedirs(DATA, exist_ok=True)
    k = key()
    if sys.argv[1] == "generate":
        tasks = []
        for level, name, desc in SUBCATS:
            for style in STYLES + SHORT:
                tasks.append({"key": f"{name}|{style}", "level": level, "subcat": name, "desc": desc, "style": style})

        def gen(t):
            res, usage = call([{"role": "user", "content": gen_prompt(t["level"], t["desc"], t["style"], 25)}],
                              GEN_SCHEMA, k, max_tokens=4000, temperature=1.0)
            return {**t, **res}, usage
        run(tasks, gen, os.path.join(DATA, "raw_gen.jsonl"), workers=16)

    elif sys.argv[1] == "label":
        msgs = {}
        for s in open(os.path.join(DATA, "raw_gen.jsonl")):
            r = json.loads(s)
            for m in r.get("messages", []):
                m = m.strip()
                if 3 <= len(m) <= 1200:
                    msgs.setdefault(m, r)
        tasks = [{"key": m, "subcat": r["subcat"], "gen_level": r["level"], "style": r["style"]} for m, r in msgs.items()]

        def label(t):
            prompt = (f"Classify this message sent to a mental wellbeing chatbot into one level.\n\n{LEVELS}\n\n"
                      f"When unsure between two levels, pick the higher one.\n\nMessage:\n<<<{t['key']}>>>")
            res, usage = call([{"role": "user", "content": prompt}], LABEL_SCHEMA, k, max_tokens=20, temperature=0)
            return {**t, **res}, usage
        run(tasks, label, os.path.join(DATA, "labels.jsonl"), workers=24)


if __name__ == "__main__":
    main()
