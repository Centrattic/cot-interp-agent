"""`word-stats` — text-statistics tools over the few-shot set.

Subcommands:
  - count <sample_id> <w1> [<w2> …]
  - tf-idf
  - compare <sample_id>
  - rank <concept> [--words "w1,w2,…"]

Backed by simple n-gram extraction + Monroe et al. 2008 log-odds with an
informative Dirichlet prior built from the combined few-shot corpus.
Intended as a cheap text-statistical complement to the SAE feature search:
the SAE tool surfaces *learned* features; word-stats surfaces *surface*
features. Many useful signals (specific phrases, conclusion markers,
hedge words) are easier to discover here than via SAE feature labels.

All four subcommands write a CSV in the current directory and print a
human-readable summary. CSVs overwrite on repeat (filename keyed by
subcommand + identifier, not auto-incremented).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

from _common import fail, get_env, load_example
from _subagent import SONNET_MODEL, SubagentError, call_simple, get_cot_prefix


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def tokenise(text: str) -> list[str]:
    """Lowercase tokenisation: contiguous word characters as tokens."""
    return _TOKEN_RE.findall(text.lower())


def extract_ngrams(text: str, ns: tuple[int, ...] = (1, 2, 3)) -> Counter:
    """Return a Counter of n-grams (space-joined strings) for n in ``ns``."""
    toks = tokenise(text)
    out: Counter = Counter()
    for n in ns:
        if n == 1:
            out.update(toks)
        else:
            for i in range(len(toks) - n + 1):
                out[" ".join(toks[i : i + n])] += 1
    return out


def log_odds_dirichlet(
    yes_counter: Counter,
    no_counter: Counter,
    *,
    alpha_0: float | None = None,
) -> dict[str, tuple[float, int, int]]:
    """Monroe et al. 2008 log-odds with informative Dirichlet prior.

    Prior for each n-gram ``w`` is ``α_w = α_0 × p_w``, where ``p_w`` is the
    empirical probability of ``w`` in the combined yes+no corpus and ``α_0``
    is the total prior strength. Shrinks toward "no class signal" for
    rare or unevenly-distributed terms.

    Returns ``{term: (z_score, y_count, n_count)}`` for every term seen in
    either class.
    """
    combined = yes_counter + no_counter
    total_combined = sum(combined.values()) or 1
    Y = sum(yes_counter.values())
    N = sum(no_counter.values())

    # α_0 default: small but nonzero; about 100 prior pseudo-tokens. Lets
    # actual data (typical few-shot has ~10K-30K tokens combined) dominate
    # while still smoothing single-occurrence flukes.
    if alpha_0 is None:
        alpha_0 = float(os.environ.get("WORD_STATS_ALPHA0", "100"))

    out: dict[str, tuple[float, int, int]] = {}
    for term, bg_count in combined.items():
        p_w = bg_count / total_combined
        a_w = alpha_0 * p_w
        y_w = yes_counter.get(term, 0)
        n_w = no_counter.get(term, 0)

        # log-odds-of-odds for term in yes vs no
        # numerator probability ratio for yes corpus: (y + α) / (Y + α_0 - y - α)
        # numerator probability ratio for no  corpus: (n + α) / (N + α_0 - n - α)
        # Guard division by zero with a tiny epsilon.
        denom_y = Y + alpha_0 - y_w - a_w
        denom_n = N + alpha_0 - n_w - a_w
        if denom_y <= 0 or denom_n <= 0:
            continue
        ratio_y = (y_w + a_w) / denom_y
        ratio_n = (n_w + a_w) / denom_n
        if ratio_y <= 0 or ratio_n <= 0:
            continue

        delta = math.log(ratio_y) - math.log(ratio_n)
        var = 1.0 / (y_w + a_w) + 1.0 / (n_w + a_w)
        z = delta / math.sqrt(var)
        out[term] = (z, y_w, n_w)
    return out


def load_few_shot_texts(env: dict) -> list[tuple[str, int, str]]:
    """Return ``[(example_id, label_int, cot_prefix)]`` for every few-shot
    example. Reads from ``$AGENT_RUN_DIR/strategy/few-shot/`` regardless of
    AGENT_TYPE (test agents access via ``--add-dir`` of strategy_dir).
    """
    fs_dir = Path(env["AGENT_RUN_DIR"]) / "strategy" / "few-shot"
    if not fs_dir.exists():
        fail(f"few-shot directory not found: {fs_dir}")
    out: list[tuple[str, int, str]] = []
    for p in sorted(fs_dir.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        text = get_cot_prefix(d) or ""
        try:
            label = int(d.get("label", -1))
        except (TypeError, ValueError):
            label = -1
        out.append((p.stem, label, text))
    return out


def doc_freq(examples: list[tuple[str, int, str]]) -> Counter:
    """Number of distinct examples each n-gram appears in."""
    df: Counter = Counter()
    for _ex, _lbl, text in examples:
        seen = set(extract_ngrams(text).keys())
        for term in seen:
            df[term] += 1
    return df


def slug(s: str, n: int = 40) -> str:
    """Filesystem-safe slug for filenames."""
    s = re.sub(r"[^\w\-]+", "_", s.lower()).strip("_")
    return s[:n] or "x"


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

def count_term(text_lower: str, term: str) -> int:
    """Count occurrences of ``term`` in ``text_lower``.

    Whole-word match for unigrams, substring (non-overlapping, left-to-right
    via ``str.count``) for multi-word phrases.
    """
    term_l = term.lower().strip()
    if not term_l:
        return 0
    if " " in term_l:
        return text_lower.count(term_l)
    # unigram → whole-word boundary
    return len(re.findall(rf"\b{re.escape(term_l)}\b", text_lower))


def cmd_count(argv: list[str]) -> int:
    if len(argv) < 2:
        fail('usage: word-stats count <sample_id> <word_or_phrase> [<word> ...]')
    sample_id = argv[0]
    terms = argv[1:]

    env = get_env()
    example = load_example(env, sample_id)
    text = get_cot_prefix(example) or ""
    text_l = text.lower()

    rows = []
    total = 0
    for term in terms:
        n = count_term(text_l, term)
        total += n
        rows.append((term, n))

    out_path = Path.cwd() / f"word_stats_count_{sample_id}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["term", "count"])
        w.writerows(rows)
        w.writerow(["__total__", total])

    print(f"status: success")
    print(f"sample_id: {sample_id}")
    print(f"terms: {len(terms)}   total: {total}")
    for term, n in rows:
        print(f"  {n:>4d}  {term}")
    print(f"details: {out_path.name}")
    return 0


# ---------------------------------------------------------------------------
# tf-idf
# ---------------------------------------------------------------------------

def cmd_tfidf(argv: list[str]) -> int:
    if argv:
        fail("usage: word-stats tf-idf  (takes no arguments)")
    env = get_env()
    examples = load_few_shot_texts(env)
    if not examples:
        fail("no few-shot examples found.")

    yes_counter: Counter = Counter()
    no_counter: Counter = Counter()
    yes_doc_freq: Counter = Counter()
    no_doc_freq: Counter = Counter()
    n_yes = n_no = 0

    for _ex, lbl, text in examples:
        grams = extract_ngrams(text)
        if lbl == 1:
            yes_counter += grams
            for g in set(grams):
                yes_doc_freq[g] += 1
            n_yes += 1
        elif lbl == 0:
            no_counter += grams
            for g in set(grams):
                no_doc_freq[g] += 1
            n_no += 1

    if n_yes == 0 or n_no == 0:
        fail(f"need both classes in few-shot; got n_yes={n_yes} n_no={n_no}")

    scores = log_odds_dirichlet(yes_counter, no_counter)
    items = sorted(scores.items(), key=lambda kv: kv[1][0])  # ascending z

    yes_top = [(t, *v, yes_doc_freq[t], no_doc_freq[t]) for t, v in items[::-1] if v[0] > 0][:20]
    no_top  = [(t, *v, yes_doc_freq[t], no_doc_freq[t]) for t, v in items     if v[0] < 0][:20]

    cwd = Path.cwd()
    yes_csv = cwd / "word_stats_tfidf_yes.csv"
    no_csv  = cwd / "word_stats_tfidf_no.csv"
    cols = ["term", "z", "y_count", "n_count", "y_examples", "n_examples"]
    for path, rows in ((yes_csv, yes_top), (no_csv, no_top)):
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for term, z, y, n, ye, ne in rows:
                w.writerow([term, f"{z:.3f}", y, n, ye, ne])

    def print_table(title: str, rows: list) -> None:
        print(f"\n=== {title} ===")
        print(f"  {'z':>6}  {'y':>4}  {'n':>4}  {'ye':>3}  {'ne':>3}  term")
        for term, z, y, n, ye, ne in rows:
            print(f"  {z:>6.2f}  {y:>4d}  {n:>4d}  {ye:>3d}  {ne:>3d}  {term}")

    print(f"status: success")
    print(f"few-shot: n_yes={n_yes} n_no={n_no}  Y_tokens={sum(yes_counter.values())} N_tokens={sum(no_counter.values())}")
    print_table(f"top 20 YES-distinctive (label=1)", yes_top)
    print_table(f"top 20 NO-distinctive (label=0)",  no_top)
    print(f"\ndetails: {yes_csv.name}, {no_csv.name}")
    return 0


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def cmd_compare(argv: list[str]) -> int:
    if len(argv) != 1:
        fail("usage: word-stats compare <sample_id>")
    sample_id = argv[0]
    env = get_env()
    example = load_example(env, sample_id)
    text = get_cot_prefix(example) or ""
    sample_grams = extract_ngrams(text)

    examples = load_few_shot_texts(env)
    yes_counter: Counter = Counter()
    no_counter: Counter = Counter()
    for _ex, lbl, t in examples:
        grams = extract_ngrams(t)
        if lbl == 1: yes_counter += grams
        elif lbl == 0: no_counter += grams

    if not yes_counter or not no_counter:
        fail("compare needs both classes present in few-shot.")

    # Restrict to terms that appear at least once in this sample.
    scores_all = log_odds_dirichlet(yes_counter, no_counter)
    scores = {t: v for t, v in scores_all.items() if t in sample_grams}

    # Top 20 by abs z, signed
    items = sorted(scores.items(), key=lambda kv: -abs(kv[1][0]))[:20]

    out_path = Path.cwd() / f"word_stats_compare_{sample_id}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["term", "z", "count_in_sample", "y_count_few_shot", "n_count_few_shot", "lean"])
        for term, (z, y, n) in items:
            lean = "yes" if z > 0 else "no"
            w.writerow([term, f"{z:.3f}", sample_grams.get(term, 0), y, n, lean])

    print(f"status: success")
    print(f"sample_id: {sample_id}  unique n-grams in sample: {len(sample_grams)}")
    print(f"\n  {'z':>6}  {'in_sample':>9}  {'y_fs':>4}  {'n_fs':>4}  lean  term")
    for term, (z, y, n) in items:
        lean = "YES" if z > 0 else "NO"
        print(f"  {z:>6.2f}  {sample_grams.get(term, 0):>9d}  {y:>4d}  {n:>4d}  {lean:>4s}  {term}")

    # Net lean across the top-20 abs-z signals
    yes_score = sum(z for _, (z, _, _) in items if z > 0)
    no_score  = sum(-z for _, (z, _, _) in items if z < 0)
    if yes_score > no_score:
        net = f"net lean: YES (Σz_pos={yes_score:.1f} vs Σ|z_neg|={no_score:.1f})"
    elif no_score > yes_score:
        net = f"net lean: NO  (Σ|z_neg|={no_score:.1f} vs Σz_pos={yes_score:.1f})"
    else:
        net = "net lean: tied"
    print(f"\n{net}")
    print(f"details: {out_path.name}")
    return 0


# ---------------------------------------------------------------------------
# rank
# ---------------------------------------------------------------------------

CONCEPT_CACHE = ".word_stats_concept_cache.json"

EXPANSION_SYSTEM = (
    "You are an expert at lexical expansion for text-search applications. "
    "Given a user concept, return 15-30 keywords and short phrases that "
    "capture variations and synonyms of the concept as it might appear in "
    "natural text. Mix unigrams (e.g. 'wait', 'reconsider') and short "
    "multi-word phrases (e.g. 'let me think', 'wait, but'). Include "
    "idiomatic phrasings and common collocations. Prefer concrete surface "
    "phrasings over abstract synonyms. Do not include the concept itself "
    "verbatim more than once."
)
EXPANSION_TOOL = "submit_keywords"
EXPANSION_SCHEMA = {
    "type": "object",
    "properties": {
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 15,
            "maxItems": 30,
            "description": "Keyword/phrase list, each between 1 and 4 words.",
        },
    },
    "required": ["keywords"],
}


def _cache_key(concept: str) -> str:
    return hashlib.sha1(f"{SONNET_MODEL}|{concept.strip().lower()}".encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    p = Path.cwd() / CONCEPT_CACHE
    if not p.exists(): return {}
    try: return json.loads(p.read_text(encoding="utf-8"))
    except Exception: return {}


def _save_cache(cache: dict) -> None:
    p = Path.cwd() / CONCEPT_CACHE
    p.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def expand_concept(concept: str) -> list[str]:
    cache = _load_cache()
    key = _cache_key(concept)
    if key in cache:
        return cache[key]
    raw = call_simple(
        system=EXPANSION_SYSTEM,
        user=f"Concept: {concept!r}\n\nExpand this concept into 15-30 keywords/phrases.",
        tool_name=EXPANSION_TOOL,
        tool_description="Submit the expanded keyword list.",
        tool_input_schema=EXPANSION_SCHEMA,
        model=SONNET_MODEL,
    )
    keywords = list(dict.fromkeys(k.strip() for k in raw.get("keywords", []) if k.strip()))
    cache[key] = keywords
    _save_cache(cache)
    return keywords


def cmd_rank(argv: list[str]) -> int:
    if not argv:
        fail('usage: word-stats rank <concept> [--words "w1,w2,..."]')
    # parse --words
    words_arg = None
    rest: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--words" and i + 1 < len(argv):
            words_arg = argv[i + 1]
            i += 2
        else:
            rest.append(argv[i])
            i += 1
    if not rest:
        fail('usage: word-stats rank <concept> [--words "w1,w2,..."]')
    concept = " ".join(rest)

    if words_arg:
        keywords = [w.strip() for w in words_arg.split(",") if w.strip()]
        source = "user-provided --words"
    else:
        try:
            keywords = expand_concept(concept)
        except SubagentError as e:
            fail(str(e), code=5)
        source = f"Codex expansion of {concept!r}"

    if not keywords:
        fail("no keywords to match.")

    env = get_env()
    examples = load_few_shot_texts(env)
    if not examples:
        fail("no few-shot examples found.")

    # Per-example: total hit count + per-keyword breakdown
    rows = []
    for ex_id, label, text in examples:
        text_l = text.lower()
        per_kw = {kw: count_term(text_l, kw) for kw in keywords}
        total = sum(per_kw.values())
        rows.append((ex_id, label, total, per_kw))
    rows.sort(key=lambda r: -r[2])

    cwd = Path.cwd()
    out_path = cwd / f"word_stats_rank_{slug(concept)}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["example_id", "label", "total_hits"] + keywords)
        for ex_id, label, total, per_kw in rows:
            w.writerow([ex_id, label, total] + [per_kw[k] for k in keywords])

    # stdout: keyword list at top, then ranked table
    print(f"status: success")
    print(f"concept: {concept!r}")
    print(f"keyword source: {source}")
    print(f"keywords ({len(keywords)}):")
    for kw in keywords:
        print(f"  - {kw}")
    print(f"\nranked few-shot (most → fewest hits):")
    print(f"  {'hits':>5}  label  example_id")
    for ex_id, label, total, _ in rows:
        print(f"  {total:>5d}  {label:>5}  {ex_id}")

    # Brief class-skew summary
    yes_total = sum(t for _, l, t, _ in rows if l == 1)
    no_total  = sum(t for _, l, t, _ in rows if l == 0)
    n_yes = sum(1 for _, l, _, _ in rows if l == 1)
    n_no  = sum(1 for _, l, _, _ in rows if l == 0)
    if n_yes and n_no:
        per_yes = yes_total / n_yes
        per_no  = no_total / n_no
        skew = "label=1 (more)" if per_yes > per_no else "label=0 (more)"
        print(
            f"\nclass means: label=1 → {per_yes:.2f} hits/example "
            f"(n={n_yes}); label=0 → {per_no:.2f} (n={n_no})  →  {skew}"
        )
    print(f"\ndetails: {out_path.name}")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

USAGE = (
    "usage: word-stats <subcommand> [args...]\n"
    "  word-stats count <sample_id> <w1> [<w2> ...]\n"
    "  word-stats tf-idf\n"
    "  word-stats compare <sample_id>\n"
    "  word-stats rank <concept> [--words \"w1,w2,...\"]"
)


def main(argv: list[str]) -> int:
    if not argv:
        fail(USAGE)
    sub, rest = argv[0], argv[1:]
    if sub == "count":  return cmd_count(rest)
    if sub == "tf-idf": return cmd_tfidf(rest)
    if sub == "compare": return cmd_compare(rest)
    if sub == "rank":   return cmd_rank(rest)
    fail(f"unknown subcommand {sub!r}\n{USAGE}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
