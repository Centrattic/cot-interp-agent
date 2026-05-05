"""`sample` tool — resample short oracle answers about an example, optionally
after a small visible-text intervention, and optionally validate the question
across the labeled few-shot pool.

Usage:
    sample <example_id> --question "<question>" [--times N] [--ans LABEL [LABEL ...]]
    sample --diff --question "<question>" [--times N] --ans POS NEG

Interventions (single-example mode only):
    --append-cot TEXT
    --prepend-cot TEXT
    --replace OLD NEW
    --around-text TEXT --insert-before TEXT
    --around-text TEXT --insert-after TEXT
    --baseline   # if an intervention is present, also run the unmodified example
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from argparse import ArgumentParser
from copy import deepcopy

from _answer_report import (
    extract_explicit_label,
    is_effectively_empty,
    normalize_label,
    normalize_response,
    summarize_responses,
)
from _common import (
    example_dir,
    fail,
    get_env,
    load_example,
    next_numbered_output_path,
    write_csv,
)


MAX_QUESTION_TOKENS = 50
MAX_RESPONSE_TOKENS = 100
DEFAULT_NUM_SAMPLES = 5
MAX_ATTEMPTS_MULTIPLIER = 3

DEFAULT_TOKENIZER = "Qwen/Qwen3-32B"
DEFAULT_MODEL = "qwen/qwen3-32b"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
EDITABLE_FIELDS = (
    "content",
    "thinking",
    "cot_prefix",
    "chain_of_thought",
    "cot_text",
    "reasoning",
    "cot_content",
    "full_response",
)


def output_prefix(default: str) -> str:
    value = os.environ.get("SAMPLE_OUTPUT_PREFIX", "").strip()
    return value or default


def get_readme_description() -> str:
    return f"""### `sample <example_id> --question "..." [--times N] [--ans LABEL [LABEL ...]]`

Resample short oracle answers about an example via OpenRouter. This is the
general answer-sampling tool: use it to ask a follow-up question about one
example, to validate a candidate question on the labeled few-shot pool with
`--diff`, or to lightly edit the visible CoT text before resampling.

**Core modes**
- `sample <example_id> --question "..."` asks one follow-up question about one example.
- Add `--ans yes no` or similar if you want explicit label aggregation.
- `sample --diff --question "..." --ans POS NEG` runs the same question over the visible labeled few-shot examples and reports whether the question separates positives from negatives.

**Visible-text intervention (single-example mode only)**
- `--append-cot TEXT`
- `--prepend-cot TEXT`
- `--replace OLD NEW`
- `--around-text TEXT --insert-before TEXT`
- `--around-text TEXT --insert-after TEXT`
- Add `--baseline` to run the unmodified example too and print the delta.

These interventions edit the **visible example text shown to the oracle**.
They do not expose hidden labels or hidden model state.

**Limits**
- Question must tokenize to **≤{MAX_QUESTION_TOKENS} Qwen tokens**.
- The reply is truncated to the **first {MAX_RESPONSE_TOKENS} tokens** (reasoning / thinking excluded). OpenRouter is also capped at {MAX_RESPONSE_TOKENS} output tokens.
- By default, the tool keeps resampling until it collects **{DEFAULT_NUM_SAMPLES} valid samples**. Override with `--times N`.
- Invalid outputs are retried automatically up to a fixed cap and still written to the details artifact.

**Scope**
- **Strategy agent:** may query any few-shot `<example_id>`.
- **Test agent:** may only query its own assigned example — `AGENT_EXAMPLE_ID` names it; other IDs are rejected.
- `--diff` is strategy-only because it relies on labeled few-shot examples.

**Interpretation warning**
- `agreement_strength` measures **consistency of sampled answers**, not correctness of your question.
- A bad proxy question can still get `agreement_strength=strong`.
- Use `--diff` to validate candidate questions on known few-shot positives and negatives before trusting them.

**Output**
- Single-example mode writes `sample_<n>.csv`.
- Diff mode writes `sample_diff_<n>.csv`.
""".strip()


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="sample",
        description="Resample short oracle answers about an example via OpenRouter.",
    )
    parser.add_argument("example_id", nargs="?")
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--times",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of valid samples to collect (default: {DEFAULT_NUM_SAMPLES}).",
    )
    parser.add_argument(
        "--ans",
        nargs="+",
        help="Optional explicit labels to aggregate, e.g. --ans yes no or --ans A B.",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Run the question over the labeled visible few-shot examples.",
    )
    parser.add_argument(
        "--append-cot",
        help="Append text to the visible CoT/reasoning field before sampling.",
    )
    parser.add_argument(
        "--prepend-cot",
        help="Prepend text to the visible CoT/reasoning field before sampling.",
    )
    parser.add_argument(
        "--replace",
        nargs=2,
        metavar=("OLD", "NEW"),
        help="Replace the first exact occurrence of OLD with NEW in the visible CoT field.",
    )
    parser.add_argument(
        "--around-text",
        help="Anchor text for a relative insertion into the visible CoT field.",
    )
    parser.add_argument(
        "--insert-before",
        help="Insert this text immediately before the last case-insensitive match of --around-text.",
    )
    parser.add_argument(
        "--insert-after",
        help="Insert this text immediately after the last case-insensitive match of --around-text.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="If an intervention is present, also sample the unmodified example and print the delta.",
    )
    return parser


def get_tokenizer():
    from tokenizers import Tokenizer  # lazy import

    name = os.environ.get("QWEN_TOKENIZER", DEFAULT_TOKENIZER)
    return Tokenizer.from_pretrained(name)


def check_test_agent_scope(env: dict, example_id: str) -> None:
    if env["AGENT_TYPE"] != "test":
        return
    assigned = os.environ.get("AGENT_EXAMPLE_ID")
    if assigned and assigned != example_id:
        fail(
            f"test agent can only query its own example "
            f"(AGENT_EXAMPLE_ID={assigned!r}), not {example_id!r}"
        )


def load_visible_labeled_examples(env: dict) -> list[tuple[str, dict]]:
    if env["AGENT_TYPE"] != "strategy":
        fail("--diff is only supported for strategy agents with labeled few-shot examples")
    base = example_dir(env)
    out: list[tuple[str, dict]] = []
    for path in sorted(base.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "label" in data:
            out.append((path.stem, data))
    return out


def call_openrouter(example_content: str, question: str, model: str) -> dict:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        fail("OPENROUTER_API_KEY not set in environment")

    system = (
        "You will be shown an example and a short follow-up question about it. "
        "Answer the question directly and concisely."
    )
    user = f"Example:\n{example_content}\n\nQuestion: {question}"

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "provider": {"only": ["alibaba"], "allow_fallbacks": False},
        "reasoning": {"enabled": False, "exclude": True},
        "max_tokens": MAX_RESPONSE_TOKENS,
    }
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        fail(f"OpenRouter HTTP {e.code}: {detail}", code=4)
    except urllib.error.URLError as e:
        fail(f"OpenRouter request failed: {e}", code=4)


def extract_content(resp: dict) -> str:
    try:
        msg = resp["choices"][0]["message"]
    except (KeyError, IndexError):
        fail(f"unexpected OpenRouter response shape: {json.dumps(resp)[:500]}")
    return (msg.get("content") or "").strip()


def validate_labels(answer_labels: list[str]) -> list[str]:
    if not answer_labels:
        return []
    normalized_labels = [normalize_label(label) for label in answer_labels]
    if any(not label.strip() for label in answer_labels):
        fail("--ans labels must be non-empty")
    if len(set(normalized_labels)) != len(normalized_labels):
        fail(f"--ans labels must be distinct, got {answer_labels!r}")
    return normalized_labels


def intervention_spec_from_args(args) -> dict[str, str] | None:
    count = sum(
        1
        for present in (
            args.append_cot is not None,
            args.prepend_cot is not None,
            args.replace is not None,
            args.around_text is not None,
        )
        if present
    )
    if count > 1:
        fail("choose at most one intervention mode")
    if args.around_text is not None:
        if bool(args.insert_before) == bool(args.insert_after):
            fail("--around-text requires exactly one of --insert-before or --insert-after")
        return {
            "kind": "insert_before" if args.insert_before is not None else "insert_after",
            "anchor": args.around_text,
            "text": args.insert_before if args.insert_before is not None else args.insert_after,
        }
    if args.append_cot is not None:
        return {"kind": "append", "text": args.append_cot}
    if args.prepend_cot is not None:
        return {"kind": "prepend", "text": args.prepend_cot}
    if args.replace is not None:
        return {"kind": "replace", "old": args.replace[0], "new": args.replace[1]}
    return None


def get_editable_field(payload: dict) -> str | None:
    for field in EDITABLE_FIELDS:
        value = payload.get(field)
        if isinstance(value, str):
            return field
    return None


def apply_intervention(payload: dict, intervention: dict[str, str]) -> tuple[dict, str]:
    updated = deepcopy(payload)
    field = get_editable_field(updated)
    if field is None:
        fail(
            "no editable visible-text field found for intervention; looked for "
            + ", ".join(EDITABLE_FIELDS)
        )
    original = updated[field]
    assert isinstance(original, str)

    kind = intervention["kind"]
    if kind == "append":
        updated[field] = original + intervention["text"]
    elif kind == "prepend":
        updated[field] = intervention["text"] + original
    elif kind == "replace":
        old = intervention["old"]
        if old not in original:
            fail(f"--replace could not find {old!r} in field {field}")
        updated[field] = original.replace(old, intervention["new"], 1)
    elif kind in ("insert_before", "insert_after"):
        anchor = intervention["anchor"]
        matches = list(re.finditer(re.escape(anchor), original, flags=re.IGNORECASE))
        if not matches:
            fail(f"--around-text could not find {anchor!r} in field {field}")
        match = matches[-1]
        if kind == "insert_before":
            updated[field] = original[: match.start()] + intervention["text"] + original[match.start() :]
        else:
            updated[field] = original[: match.end()] + intervention["text"] + original[match.end() :]
    else:
        fail(f"unsupported intervention kind {kind!r}")

    return updated, field


def render_example_content(example: dict) -> str:
    payload = example.get("content")
    if payload is None:
        payload = {k: v for k, v in example.items() if k != "label"}
    return payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)


def collect_samples(
    *,
    example_id: str,
    example_content: str,
    question: str,
    q_count: int,
    model: str,
    tok,
    num_samples: int,
    answer_labels: list[str],
    variant: str,
) -> tuple[list[dict[str, str | int]], list[str], list[str], list[str], int, int]:
    rows: list[dict[str, str | int]] = []
    truncated_responses: list[str] = []
    normalized_responses: list[str] = []
    parsed_labels: list[str] = []
    attempts = 0
    invalid_attempts = 0
    max_attempts = num_samples * MAX_ATTEMPTS_MULTIPLIER

    while len(truncated_responses) < num_samples and attempts < max_attempts:
        attempts += 1
        resp = call_openrouter(example_content, question, model)
        raw = extract_content(resp)

        resp_ids = tok.encode(raw).ids
        truncated_ids = resp_ids[:MAX_RESPONSE_TOKENS]
        truncated = tok.decode(truncated_ids)
        normalized = normalize_response(truncated)
        parsed_label = extract_explicit_label(truncated, answer_labels) if answer_labels else None

        invalid_reason = ""
        if is_effectively_empty(truncated):
            is_valid = False
            invalid_reason = "empty_response"
        elif answer_labels and parsed_label is None:
            is_valid = False
            invalid_reason = "unparseable_label"
        else:
            is_valid = True

        if is_valid:
            truncated_responses.append(truncated)
            normalized_responses.append(normalized)
            if parsed_label is not None:
                parsed_labels.append(parsed_label)
        else:
            invalid_attempts += 1

        rows.append(
            {
                "variant": variant,
                "status": "success" if is_valid else "invalid",
                "example_id": example_id,
                "question": question,
                "question_tokens": q_count,
                "model": model,
                "num_samples": num_samples,
                "attempt_index": attempts,
                "sample_index": len(truncated_responses) if is_valid else "",
                "is_valid": str(is_valid).lower(),
                "invalid_reason": invalid_reason,
                "parsed_label": parsed_label or "",
                "response": truncated,
                "response_normalized": normalized,
                "response_tokens": len(truncated_ids),
                "response_raw": raw,
            }
        )

    return rows, truncated_responses, normalized_responses, parsed_labels, attempts, invalid_attempts


def print_summary_line(summary: dict, *, answer_labels: list[str], num_samples: int, prefix: str) -> None:
    if answer_labels:
        print(
            f"{prefix}: "
            f"majority_label={summary['majority_label']!r}; "
            f"runner_up_label={summary['runner_up_label']!r}; "
            f"majority_label_count={summary['majority_label_count']}/{num_samples}; "
            f"majority_label_rate={summary['majority_label_rate']:.2f}; "
            f"vote_margin={summary['vote_margin']:.2f}; "
            f"valid_samples={summary['valid_samples']}/{num_samples}; "
            f"invalid_attempts={summary['invalid_attempts']}; "
            f"invalid_attempt_rate={summary['invalid_attempt_rate']:.2f}; "
            f"unique_labels={summary['unique_labels']}; "
            f"label_entropy={summary['label_entropy']:.2f}; "
            f"agreement_strength={summary['strength']}"
        )
        return
    print(
        f"{prefix}: "
        f"majority_response={summary['majority_response']!r}; "
        f"runner_up_response={summary['runner_up_response']!r}; "
        f"majority_response_count={summary['majority_response_count']}/{num_samples}; "
        f"majority_response_rate={summary['majority_response_rate']:.2f}; "
        f"vote_margin={summary['vote_margin']:.2f}; "
        f"valid_samples={summary['valid_samples']}/{num_samples}; "
        f"invalid_attempts={summary['invalid_attempts']}; "
        f"invalid_attempt_rate={summary['invalid_attempt_rate']:.2f}; "
        f"unique_responses={summary['unique_responses']}; "
        f"response_entropy={summary['response_entropy']:.2f}; "
        f"agreement_strength={summary['strength']}"
    )


def run_single_example(args, env: dict, tok, normalized_labels: list[str], intervention: dict[str, str] | None) -> int:
    if not args.example_id:
        fail("example_id is required unless you use --diff")
    example_id = args.example_id
    check_test_agent_scope(env, example_id)
    example = load_example(env, example_id)

    model = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    q_count = len(tok.encode(args.question).ids)
    details_path = next_numbered_output_path(output_prefix("sample"))

    if q_count > MAX_QUESTION_TOKENS:
        write_csv(
            details_path,
            ["status", "example_id", "question", "question_tokens", "invalid_reason"],
            [
                {
                    "status": "failed",
                    "example_id": example_id,
                    "question": args.question,
                    "question_tokens": q_count,
                    "invalid_reason": f"question_too_long(limit={MAX_QUESTION_TOKENS})",
                }
            ],
        )
        print(
            f"status: failed -- question is {q_count} tokens "
            f"(limit {MAX_QUESTION_TOKENS})"
        )
        print(f"details: {details_path.name}")
        return 1

    variants: list[tuple[str, str]] = []
    base_content = render_example_content(example)
    if args.baseline and intervention is None:
        fail("--baseline only makes sense with an intervention")
    if args.baseline:
        variants.append(("baseline", base_content))

    if intervention is not None:
        payload = example.get("content")
        if payload is None:
            payload = {k: v for k, v in example.items() if k != "label"}
        if isinstance(payload, str):
            payload = {"content": payload}
        edited_payload, edited_field = apply_intervention(payload, intervention)
        edited_content = render_example_content({"content": edited_payload["content"]} if list(edited_payload.keys()) == ["content"] else edited_payload)
        variants.append(("intervention", edited_content))
    else:
        variants.append(("sample", base_content))
        edited_field = None

    all_rows: list[dict[str, str | int]] = []
    results: dict[str, dict] = {}
    outputs: dict[str, list[str]] = {}

    for variant_name, content in variants:
        rows, truncated_responses, normalized_responses, parsed_labels, attempts, invalid_attempts = collect_samples(
            example_id=example_id,
            example_content=content,
            question=args.question,
            q_count=q_count,
            model=model,
            tok=tok,
            num_samples=args.times,
            answer_labels=args.ans or [],
            variant=variant_name,
        )
        all_rows.extend(rows)
        outputs[variant_name] = truncated_responses
        if len(truncated_responses) < args.times:
            write_csv(details_path, list(all_rows[0].keys()), all_rows)
            print(
                "status: failed -- "
                f"variant={variant_name}; collected {len(truncated_responses)}/{args.times} "
                f"valid responses after {attempts} attempts"
            )
            print(f"details: {details_path.name}")
            return 1
        results[variant_name] = summarize_responses(
            normalized_responses,
            parsed_labels=parsed_labels if args.ans else None,
            invalid_attempts=invalid_attempts,
        )

    write_csv(details_path, list(all_rows[0].keys()), all_rows)

    print("status: success")
    if intervention is not None:
        print(f"intervention: {intervention['kind']}")
        if edited_field is not None:
            print(f"editable_field: {edited_field}")
    if args.baseline and "baseline" in results and "intervention" in results:
        print_summary_line(results["baseline"], answer_labels=args.ans or [], num_samples=args.times, prefix="baseline")
        print_summary_line(results["intervention"], answer_labels=args.ans or [], num_samples=args.times, prefix="intervention")
        if args.ans:
            majority_changed = results["baseline"]["majority_label"] != results["intervention"]["majority_label"]
        else:
            majority_changed = results["baseline"]["majority_response"] != results["intervention"]["majority_response"]
        print(
            "delta: "
            f"majority_changed={str(majority_changed).lower()}; "
            f"vote_margin_shift={results['intervention']['vote_margin'] - results['baseline']['vote_margin']:.2f}; "
            f"invalid_attempt_rate_shift={results['intervention']['invalid_attempt_rate'] - results['baseline']['invalid_attempt_rate']:.2f}"
        )
        for variant_name in ("baseline", "intervention"):
            print(f"{variant_name}_responses:")
            for sample_index, response in enumerate(outputs[variant_name], start=1):
                print(f"{sample_index}: {response}")
    else:
        sole_variant = variants[0][0]
        print_summary_line(results[sole_variant], answer_labels=args.ans or [], num_samples=args.times, prefix="summary")
        print("responses:")
        for sample_index, response in enumerate(outputs[sole_variant], start=1):
            print(f"{sample_index}: {response}")
    print(f"details: {details_path.name}")
    return 0


def coerce_binary_label(value) -> int | None:
    if value in (0, 1):
        return int(value)
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        return int(value.strip())
    return None


def run_diff(args, env: dict, tok) -> int:
    if len(args.ans or []) != 2:
        fail("--diff currently requires exactly two explicit labels via --ans")

    model = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    q_count = len(tok.encode(args.question).ids)
    details_path = next_numbered_output_path(output_prefix("sample_diff"))

    if q_count > MAX_QUESTION_TOKENS:
        write_csv(
            details_path,
            ["status", "question", "question_tokens", "invalid_reason"],
            [
                {
                    "status": "failed",
                    "question": args.question,
                    "question_tokens": q_count,
                    "invalid_reason": f"question_too_long(limit={MAX_QUESTION_TOKENS})",
                }
            ],
        )
        print(
            f"status: failed -- question is {q_count} tokens "
            f"(limit {MAX_QUESTION_TOKENS})"
        )
        print(f"details: {details_path.name}")
        return 1

    labeled = load_visible_labeled_examples(env)
    if not labeled:
        fail("no labeled visible examples available for --diff")

    positive_answer = args.ans[0]
    negative_answer = args.ans[1]
    rows: list[dict[str, str | int | float]] = []
    tp = tn = fp = fn = skipped = 0
    pos_examples = neg_examples = 0

    for example_id, example in labeled:
        true_label = coerce_binary_label(example.get("label"))
        if true_label is None:
            skipped += 1
            continue
        example_content = render_example_content(example)
        attempt_rows, truncated_responses, normalized_responses, parsed_labels, attempts, invalid_attempts = collect_samples(
            example_id=example_id,
            example_content=example_content,
            question=args.question,
            q_count=q_count,
            model=model,
            tok=tok,
            num_samples=args.times,
            answer_labels=args.ans or [],
            variant="diff",
        )
        if len(truncated_responses) < args.times:
            skipped += 1
            rows.append(
                {
                    "example_id": example_id,
                    "true_label": true_label,
                    "status": "failed",
                    "majority_label": "",
                    "runner_up_label": "",
                    "vote_margin": "",
                    "agreement_strength": "",
                    "valid_samples": len(truncated_responses),
                    "invalid_attempts": invalid_attempts,
                    "invalid_attempt_rate": invalid_attempts / max(1, attempts),
                    "responses_json": json.dumps(truncated_responses, ensure_ascii=False),
                    "parsed_labels_json": json.dumps(parsed_labels, ensure_ascii=False),
                }
            )
            continue

        summary = summarize_responses(
            normalized_responses,
            parsed_labels=parsed_labels,
            invalid_attempts=invalid_attempts,
        )
        predicted_positive = summary["majority_label"] == positive_answer
        if true_label == 1:
            pos_examples += 1
            if predicted_positive:
                tp += 1
            else:
                fn += 1
        else:
            neg_examples += 1
            if predicted_positive:
                fp += 1
            else:
                tn += 1
        rows.append(
            {
                "example_id": example_id,
                "true_label": true_label,
                "status": "success",
                "majority_label": summary["majority_label"],
                "runner_up_label": summary["runner_up_label"],
                "vote_margin": summary["vote_margin"],
                "agreement_strength": summary["strength"],
                "valid_samples": summary["valid_samples"],
                "invalid_attempts": summary["invalid_attempts"],
                "invalid_attempt_rate": summary["invalid_attempt_rate"],
                "responses_json": json.dumps(truncated_responses, ensure_ascii=False),
                "parsed_labels_json": json.dumps(parsed_labels, ensure_ascii=False),
            }
        )

    write_csv(details_path, list(rows[0].keys()), rows)

    usable = tp + tn + fp + fn
    if usable == 0 or pos_examples == 0 or neg_examples == 0:
        print(
            "status: failed -- "
            f"usable={usable}; pos_examples={pos_examples}; neg_examples={neg_examples}; skipped={skipped}"
        )
        print(f"details: {details_path.name}")
        return 1

    tpr = tp / pos_examples if pos_examples else 0.0
    tnr = tn / neg_examples if neg_examples else 0.0
    acc = (tp + tn) / usable if usable else 0.0
    pos_yes_rate = tp / pos_examples if pos_examples else 0.0
    neg_yes_rate = fp / neg_examples if neg_examples else 0.0

    print("status: success")
    print(
        "diff: "
        f"positive_answer={positive_answer!r}; negative_answer={negative_answer!r}; "
        f"n={usable}; skipped={skipped}; "
        f"pos_majority_{normalize_label(positive_answer)}={tp}/{pos_examples}; "
        f"neg_majority_{normalize_label(positive_answer)}={fp}/{neg_examples}; "
        f"gap={pos_yes_rate - neg_yes_rate:.2f}"
    )
    print(
        "classifier: "
        f"TP={tp}; TN={tn}; FP={fp}; FN={fn}; "
        f"acc={acc:.3f}; TPR={tpr:.3f}; TNR={tnr:.3f}; gmean2={tpr * tnr:.3f}"
    )
    print(f"details: {details_path.name}")
    return 0


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    question = args.question.strip()
    if not question:
        fail("question must be non-empty")
    if args.times < 1:
        fail(f"--times must be >= 1, got {args.times}")
    if args.diff and args.example_id:
        fail("do not pass example_id together with --diff")

    env = get_env()
    tok = get_tokenizer()
    validate_labels(args.ans or [])
    intervention = intervention_spec_from_args(args)

    if args.diff:
        if intervention is not None:
            fail("--diff does not currently support interventions")
        if args.baseline:
            fail("--baseline is not supported with --diff")
        return run_diff(args, env, tok)
    return run_single_example(args, env, tok, [normalize_label(label) for label in args.ans or []], intervention)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
