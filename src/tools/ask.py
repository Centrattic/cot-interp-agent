"""Compatibility wrapper for the legacy `ask` CLI.

The implementation lives in `sample.py`; `ask` keeps the older positional
syntax:

    ask <example_id> "<question>" [--times N] [--ans LABEL [LABEL ...]]
"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser

from sample import main as sample_main

DEFAULT_NUM_SAMPLES = 5


def get_readme_description() -> str:
    return (
        "### `ask <example_id> \"<question>\" [--times N] [--ans LABEL [LABEL ...]]`\n\n"
        "Legacy wrapper for `sample`. Prefer `sample` for new work: it supports the same"
        " single-example oracle sampling plus `--diff` and visible-text interventions."
    )


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="ask",
        description="Ask a short follow-up question about an example via OpenRouter.",
    )
    parser.add_argument("example_id")
    parser.add_argument("question")
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
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    os.environ["SAMPLE_OUTPUT_PREFIX"] = "ask"
    forwarded = [args.example_id, "--question", args.question, "--times", str(args.times)]
    if args.ans:
        forwarded.extend(["--ans", *args.ans])
    return sample_main(forwarded)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
