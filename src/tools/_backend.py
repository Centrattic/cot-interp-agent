"""Model inference backend for interpretability tools.

Four call sites the tool CLIs dispatch into:
  - force_tokens(example, position, forced_tokens) -> next_token_str
  - get_logit(example, position, token) -> float
  - get_entropy(example, position, tokens) -> float
  - ask_model(example, question, max_response_tokens) -> response_str

Tokenization / generation limits are enforced by the CLI layer, not here.
This module is a single place to wire in whichever model produced the
pre-computed activations under data/<task>/{few-shot,test}/*.npy.

To implement, replace the NotImplementedError bodies with real inference.
The existing tokenize_count() helper is used by the CLI to enforce the
10-token-question limit on `ask`, so you must supply a real tokenizer to
take the tools out of stub mode.
"""

from __future__ import annotations


class BackendNotConfigured(RuntimeError):
    """Raised when a tool is invoked before a model backend has been wired in."""


def _stub(name: str):
    raise BackendNotConfigured(
        f"{name}: no model backend wired in. "
        f"Implement src/tools/_backend.py against the model that generated "
        f"data/<task>/*.npy (tokenizer + forward pass with logits)."
    )


def tokenize_count(text: str) -> int:
    """Return the number of tokens `text` tokenizes to.

    Used by CLI wrappers to enforce input-length limits (force ≤10 tokens,
    ask ≤10-token question). Stub returns a conservative whitespace count.
    Replace with the real tokenizer before running against live data.
    """
    # Conservative stand-in until a real tokenizer is wired in.
    return len(text.split())


def force_tokens(example: dict, position: int, forced_tokens: str) -> str:
    """Run the model on the example with `forced_tokens` spliced in at
    `position`, and return the single next token the model predicts."""
    _stub("force_tokens")


def get_logit(example: dict, position: int, token: str) -> float:
    """Return the logit assigned to `token` at `position` in `example`."""
    _stub("get_logit")


def get_entropy(example: dict, position: int, tokens: list[str]) -> float:
    """Return the entropy of the model's distribution over `tokens`
    at `position` in `example`."""
    _stub("get_entropy")


def ask_model(example: dict, question: str, max_response_tokens: int) -> str:
    """Append `question` to the example, run the model, return the first
    `max_response_tokens` tokens of the response as a string."""
    _stub("ask_model")
