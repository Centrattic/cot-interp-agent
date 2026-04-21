"""Deprecated. Replaced by `top_10_logits`."""

import sys

print(
    "error: `logit` has been replaced by `top_10_logits <example_id> <token_position>`.",
    file=sys.stderr,
)
sys.exit(2)
