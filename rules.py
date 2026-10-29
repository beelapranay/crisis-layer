"""Tier 1 of the crisis layer: regex rules, no model.

check_input(text) rates a user message as "none", "checkin" or "crisis" and says which rules fired.
check_output(reply, level) decides whether a model reply may be shown: it blocks unknown phone numbers, method and
dose details, very-low-calorie plans, validation of self-harm, and goodbye letters written for an at-risk user.
strip_resources(reply) removes every sentence that names a crisis resource; the app shows its own verified card.

Rules only raise the level; nothing here lowers it. Stdlib only, so the patterns port directly to Swift
(NSRegularExpression accepts the same syntax).
"""
import re

LEVELS = ("none", "checkin", "crisis")


def _norm(text):
    return re.sub(r"\s+", " ", text.replace("’", "'").replace("‘", "'")).strip().lower()


def _compile(rules):
    return [(name, re.compile(pat)) for name, pat in rules]


# Plain intent, plans, means, goodbyes, self-harm, abuse, method requests: show the crisis screen.
CRISIS = _compile([
    ("intent", r"\b(kill|hang|shoot|drown)(ing)? (my ?self|me)\b"),
    ("intent", r"\bsuicid"),
    ("intent", r"\b(end|ending|take|taking) (my (own )?life|it all|things)\b"),
    ("intent", r"\b(don't|do not|dont) want to (be alive|live|exist|be here)( anymore)?\b"),
    ("intent", r"\bwant(ed)? to die\b|\bbetter off dead\b|\bready to die\b"),
    ("intent", r"\b(stop|stopped) existing\b|\bdisappear forever\b"),
    ("intent", r"\b(never|not) (wake|waking) up\b|\bsleep and (never|not) (wake|deal)"),
    ("intent", r"\b(give|giving) up for good\b"),
    ("intent", r"\bwon't be (around|here|alive|a problem)\b|\bnot be around (next|much|for)"),
    ("intent", r"\bdon't see myself making it\b|\bwon't make it to\b"),
    ("plan", r"\b(have|made|got) a plan\b|\btonight's the night\b|\btonight is the night\b"),
    ("plan", r"\b(do|doing) something stupid\b|\bscared of what i might do\b|\bdanger to myself\b"),
    ("plan", r"\bcan't promise i'll be safe\b|\bnot (feel )?safe (tonight|right now)\b"),
    ("means", r"\b(bought|got|have|found|buying) (a |some )?(rope|noose|gun|razor|blade)s?\b"),
    ("means", r"\b(dad's|mom's|my) gun\b"),
    ("means", r"\b(saving|saved|stockpil\w*|hoard\w*|collecting) (up )?(my |all my )?(pills|meds|medication)"),
    ("means", r"\b(took|taken|swallowed) (a lot of|all|too many|a bunch of|the whole|some) (pills|meds|tablets)"),
    ("means", r"\balready took (something|them|it|pills)\b|\boverdos"),
    ("means", r"\bhow much it would take\b|\bknow (exactly )?how much it would take\b"),
    ("place", r"\b(standing|sitting|up) on (a|the) (bridge|roof|ledge|edge)\b"),
    ("place", r"\bhow (high|tall) (the |that |this )?\w+( \w+)? is\b|\btallest (bridge|building)\b"),
    ("place", r"\b(crash|crashing|drive|driving|drove) (my )?car (into|off)|\bcrash(ing)? my car\b"),
    ("goodbye", r"\bgoodbye (note|letter|message)s?\b|\bsaid goodbye to (my |all my )?(friends|family|everyone)"),
    ("goodbye", r"\b(after|when) i'm gone\b|\bfor after i'?m gone\b"),
    ("goodbye", r"\b(updated|wrote|made|finished) my will\b"),
    ("goodbye", r"\bgiving away my (stuff|things|belongings)|\bgave (away )?my (dog|cat|pet)"),
    ("goodbye", r"\bwon't need (it|them|my \w+)\b|\bleave forever\b|\bwon't be able to look after\b"),
    ("selfharm", r"\b(cut|cutting|burn|burned|burnt|burning|hurt|hurting|harm|harming) (my ?self|my arms?|my legs?)\b"),
    ("selfharm", r"\bself[- ]?harm|\bcut deeper\b|\buntil (they|it) bleeds?\b"),
    ("abuse", r"\b(he|she|they|partner|boyfriend|girlfriend|husband|wife|dad|mom|father|mother|someone)( \w+)? "
              r"(hits|beats|chokes|locks|hurts|is hurting|keeps hurting) me\b"),
    ("abuse", r"\b(locks|locked) me in\b|\bsexually assault|\braped\b|\bscared to go home\b"),
    ("abuse", r"\bsomeone (at home )?is hurting me\b"),
    ("psychosis", r"\bvoices? (telling|tell|tells|saying) me\b|\bdeserve to die\b"),
    ("others", r"\b(hurt|harm|kill)(ing)? (my )?(baby|kids?|child|children|someone)\b"),
    ("ed", r"\b(haven't|have not|not) eaten (in|for) (\d+|two|three|four|five|six|seven) days\b"),
    ("ed", r"\b(throw|threw|thrown|throwing) up after\b|\bmake myself (throw up|vomit|sick|pass out)\b|\bpurg"),
    ("ed", r"\blost \d+ (pounds|lbs|kg) in a (week|month)\b.*\bnot eating\b|\bby not eating\b"),
    ("ed", r"\b([1-9]\d{1,2}|1[01]\d\d)[- ]?(calorie|kcal|cal)\b|\bunder ([1-9]\d{1,2}|1[01]\d\d) (calories|kcal)\b"),
    ("ed", r"\bstarv(e|ing) (myself|without)\b|\bbest way to starve\b|\bfast for (\w+ )?(weeks|a month)\b"),
    ("method", r"\blethal dose\b|\bpainless(est)? way\b|\b(ways?|how) to (die|kill myself)\b"),
    ("method", r"\b(how many|how much) .{0,40}\b(to|would it take to) (die|not wake|stop my heart|black ?out|pass out)"),
    ("method", r"\b(most effective|best|easiest) ways? to (self[- ]?harm|hurt myself|die)\b"),
    ("method", r"\b(dangerous|toxic|deadly) (to mix|if you (breathe|drink|take))\b|\bdrugs are dangerous to mix\b"),
    ("method", r"\blook like an accident\b"),
    ("method", r"\bhow long can (a person|someone|i|you) (go|survive|live) without (water|food|eating|sleep)\b"),
    ("method", r"\bstay awake for (\w+ )?days\b|\bwithout sleeping on purpose\b"),
    ("concealment", r"\bhide (my |the |that i'm )?(cuts|scars|injuries|self-harm|not eating|that i'm not eating)"),
    ("concealment", r"\b(stop|keep) my (therapist|doctor|parents|family) from finding out\b"),
    ("concealment", r"\bconvince my doctor i'm fine\b|\bwithout my \w+ noticing\b"),
    ("means", r"\b(get|getting) (my hands on )?more of my (\w+ )?(meds|pills|medication)|\bmy mom's pain pills\b"),
])

# Indirect or ambiguous signs: the model replies, and the app adds a gentle safety check.
CHECKIN = _compile([
    ("hopeless", r"\bwhat's the point( of (living|life|anything|it all))?\b|\bno reason (for me )?to (stay|live)\b"),
    ("hopeless", r"\b(no|don't see a|can't see a) way out\b|\bliving feels optional\b|\bpointless\b"),
    ("hopeless", r"\bwant (it|this|the pain)( all)? to (stop|end)\b|\bjust want it (all )?to (stop|end)\b"),
    ("hopeless", r"\bcan't (do|take|go on with) (this|it) anymore\b|\bcan't go on\b"),
    ("hopeless", r"\btired of (fighting|living|being alive|trying)\b|\bready to give up\b"),
    ("burden", r"\bbetter off without me\b|\b(i'm|i am|being) a burden\b|\bwon't be a problem\b"),
    ("burden", r"\bnobody would (notice|care|miss|come)\b|\bno one would (notice|care|miss|come)\b"),
    ("burden", r"\bif i (died|was gone|disappeared)\b|\binsurance money\b"),
    ("death", r"\b(my own|my) funeral\b|\bdisappear\b|\bstop existing\b|\bnot have to start again\b"),
    ("death", r"\btake a break from being me\b|\bdon't want to wake up\b|\bnot waking up\b"),
    ("worth", r"\b(i'm|i am) worthless\b|\btell me (i'm|i am) worthless\b|\beveryone hates me\b"),
    ("worth", r"\bpunish (my ?self|my body)\b|\bdeserve to (hear it|suffer|be hurt)\b"),
    ("substance", r"\bblack ?out\b|\bdrinking (all day|more than usual|to cope)\b"),
    ("substance", r"\bmix(ing)? (my )?(meds|pills|medication)( with alcohol)?\b|\bnumb (out|myself)\b"),
    ("meds", r"\b(stop|stopped|quit|quitting) (taking )?my (meds|medication|antidepressants)\b|\bcold turkey\b"),
    ("ed", r"\blose \d+ (pounds|lbs|kg) in a week\b|\bfastest way to lose weight\b|\beven if it's unhealthy\b"),
    ("ed", r"\bnot eating\b|\bfeel (huge|disgusting|fat)\b"),
    ("help", r"\bno point in getting help\b|\bconvince me there's no point\b"),
    ("breath", r"\bhold my breath\b"),
    ("danger", r"\bdrive .{0,30}after (taking|drinking|a few)\b"),
    # ambiguous on its own ("I don't think I'll be around for dinner"), so a check-in rather than the crisis screen
    ("intent", r"\b(don't think|not sure|doubt) i'll (be around|be here|still be alive|make it)\b"),
])


def check_input(text):
    """Returns (level, [rule names that fired])."""
    t = _norm(text)
    hits = [name for name, rx in CRISIS if rx.search(t)]
    if hits:
        return "crisis", hits
    hits = [name for name, rx in CHECKIN if rx.search(t)]
    return ("checkin", hits) if hits else ("none", [])


# Crisis resources the app trusts; any other phone-like number in a reply is treated as made up.
ALLOWED_NUMBERS = {"988", "911", "112", "999", "741741", "116123", "18002738255", "8002738255", "18006564673"}
_PHONE = re.compile(r"(?<![\w.])(\+?\d[\d\s().-]{1,14}\d)(?![\w.])")

OUTPUT_BLOCK = _compile([
    ("validates_harm", r"\b(it's|it is) (okay|ok|fine|normal) to (hurt|cut|harm|starve|punish) (yourself|your body)"),
    ("validates_harm", r"\b(it's|it is) (okay|ok) to hurt yourself\b|\byou deserve to (die|suffer)\b"),
    ("low_calorie", r"\b([1-9]\d{1,2}|1[01]\d\d)[- ]?(calorie|kcal|cal)s?\b(?! (deficit|more|extra|above|over))"),
    ("low_calorie", r"\beating under ([1-9]\d{1,2}|1[01]\d\d)\b.*\b(doable|achievable|possible)\b"),
    ("dose", r"\b\d{2,5} ?(mg|milligrams|grams|g)\b.{0,60}\b(lethal|fatal|overdose|toxic|deadly)\b"),
    ("dose", r"\b(lethal|fatal|toxic) (dose|amount)\b.{0,60}\b\d{2,5} ?(mg|milligrams|grams|g|pills|tablets)\b"),
    ("method", r"\b(standard drinks?|shots?) (to|will) (black ?out|pass out)\b"),
    ("concealment", r"\b(long sleeves|bracelets?|makeup|concealer|band-?aids?)\b.{0,80}\b(cover|hide)\b.{0,40}"
                    r"\b(cuts|scars|marks|injuries)\b"),
    ("concealment", r"\b(cover|hide) (up )?(the |your )?(cuts|scars|marks|injuries)\b"),
])
# Sentences that name or describe a crisis resource. The model garbles these ("988 Suicide & Hospice Pavilion"), so
# the app drops them from every reply and shows its own verified resource card instead.
_RESOURCE = re.compile(r"\b(988|911|741741|hotline|helpline|lifeline|crisis (line|text|center|service)|samaritans|"
                       r"emergency (services|number|room)|text line|warmline)\b", re.I)
_SENTENCE = re.compile(r"[^.!?\n]*[.!?]+[\"')\]]*|[^.!?\n]+$|\n", re.M)


def strip_resources(reply):
    """Returns (reply without sentences that mention crisis resources, whether any were removed)."""
    parts = _SENTENCE.findall(reply)
    kept = [p for p in parts if not _RESOURCE.search(p)]
    out = re.sub(r"\n{3,}", "\n\n", "".join(kept)).strip()
    return out, len(kept) < len(parts)


_LETTER = re.compile(r"\bdear \[?\w+")


def check_output(reply, level="none"):
    """Returns (ok, [reasons]). level is check_input's rating of the user message the reply answers."""
    t = _norm(reply)
    reasons = [name for name, rx in OUTPUT_BLOCK if rx.search(t)]
    for m in _PHONE.finditer(reply):
        digits = re.sub(r"\D", "", m.group(1))
        if digits in ALLOWED_NUMBERS:
            continue
        before = _norm(reply[max(0, m.start() - 25):m.start()])
        if len(digits) >= 7 or (len(digits) >= 3 and re.search(r"\b(call|text|dial|at|line)\W*$", before)):
            reasons.append(f"unknown_number:{m.group(1).strip()}")
    if level != "none" and _LETTER.search(t):
        reasons.append("goodbye_letter")
    return (not reasons), reasons
