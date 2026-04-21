"""Model inference backend for interpretability tools.

Read-only (precomputed sidecar) call sites — used by `top_10_logits` and
`top10_entropy`:
  - get_top_10_logits(env, example_id, example, position) -> list[(token, logit)]
  - get_top10_entropy(env, example_id, example, position) -> float

Both read from `<example_id>.logits.npz` written by src/precompute_logits.py.
Positions are CoT-relative (position 0 == first cot_prefix token).

Live-model call site — used by `force`:
  - force_and_next_top10(env, example_id, example, cot_position, forced_text,
                         max_forced_tokens) -> (next_token_str, [(tok, lp)×10])
Backed by Tinker's SamplingClient; requires the tinker SDK on PYTHONPATH and
TINKER_BASE_MODEL set in the environment.

Stub call site — `ask` still routes through OpenRouter directly in ask.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from _common import example_dir
from _task_io import build_prompt_parts, load_task_meta


class BackendNotConfigured(RuntimeError):
    """Raised when a tool is invoked before a model backend has been wired in."""


# ---------------------------------------------------------------------------
# Precomputed logits (read-only path) — top_10_logits / top10_entropy
# ---------------------------------------------------------------------------

def _logits_path(env: dict, example_id: str) -> Path:
    return example_dir(env) / f"{example_id}.logits.npz"


def _load_logits(env: dict, example_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (top_tokens, top_logits) arrays of shape (N, 10) each."""
    path = _logits_path(env, example_id)
    if not path.exists():
        raise BackendNotConfigured(
            f"precomputed logits not found: {path}. "
            f"Run `python src/precompute_logits.py --task {env['AGENT_TASK']}` "
            f"to populate <example_id>.logits.npz sidecars."
        )
    # top_tokens is saved with dtype=object (variable-length strings), which
    # requires pickle. Files are produced by our own precompute_logits.py.
    npz = np.load(path, allow_pickle=True)
    top_tokens = npz["top_tokens"]
    top_logits = npz["top_logits"]
    if top_tokens.shape != top_logits.shape or top_tokens.shape[1] != 10:
        raise BackendNotConfigured(
            f"logits file {path} has unexpected shape "
            f"(tokens={top_tokens.shape}, logits={top_logits.shape}); "
            f"expected (N, 10) for both."
        )
    return top_tokens, top_logits


def _position_slice(
    top_tokens: np.ndarray, top_logits: np.ndarray, position: int, example_id: str
) -> tuple[np.ndarray, np.ndarray]:
    n = top_tokens.shape[0]
    if position < 0 or position >= n:
        raise BackendNotConfigured(
            f"position {position} out of range for {example_id} "
            f"(have {n} CoT token positions)."
        )
    return top_tokens[position], top_logits[position]


def get_top_10_logits(
    env: dict, example_id: str, example: dict, position: int
) -> list[tuple[str, float]]:
    """Return the top-10 (token, logit) pairs at `position`, descending."""
    top_tokens, top_logits = _load_logits(env, example_id)
    toks, logs = _position_slice(top_tokens, top_logits, position, example_id)
    return [(str(t), float(l)) for t, l in zip(toks, logs)]


def get_top10_entropy(
    env: dict, example_id: str, example: dict, position: int
) -> float:
    """Entropy (nats) of the softmaxed top-10 distribution at `position`.

    Softmax over only the top-10 logits (shifted by max for stability). We
    don't have the full vocab, so this is an underestimate of true entropy
    but preserves the monotone ordering (peaked vs spread) we care about.
    """
    top_tokens, top_logits = _load_logits(env, example_id)
    _, logs = _position_slice(top_tokens, top_logits, position, example_id)
    logs = np.asarray(logs, dtype=np.float64)
    logs = logs - logs.max()
    probs = np.exp(logs)
    probs = probs / probs.sum()
    nz = probs[probs > 0]
    return float(-(nz * np.log(nz)).sum())


# ---------------------------------------------------------------------------
# Tinker SamplingClient — force
# ---------------------------------------------------------------------------

_SAMPLING_CLIENT = None
_SAMPLING_TOKENIZER = None


def _get_sampling_client():
    """Create (and cache) a Tinker SamplingClient for this process."""
    global _SAMPLING_CLIENT, _SAMPLING_TOKENIZER
    if _SAMPLING_CLIENT is not None:
        return _SAMPLING_CLIENT, _SAMPLING_TOKENIZER

    try:
        import tinker
    except ImportError as e:
        raise BackendNotConfigured(
            f"tinker SDK not importable ({e}); install with `uv pip install tinker`."
        )

    base_model = os.environ.get("TINKER_BASE_MODEL", "Qwen/Qwen3-32B")

    service = tinker.ServiceClient()
    _SAMPLING_CLIENT = service.create_sampling_client(base_model=base_model)
    _SAMPLING_TOKENIZER = _SAMPLING_CLIENT.get_tokenizer()
    return _SAMPLING_CLIENT, _SAMPLING_TOKENIZER


def _split_for_agent(env: dict, split_of: dict[str, str]) -> str:
    """Map the agent's workspace type to the cot-proxy-tasks source split."""
    return split_of["few-shot" if env["AGENT_TYPE"] == "strategy" else "test"]


def force_and_next_top10(
    *,
    env: dict,
    example_id: str,
    example: dict,
    cot_position: int,
    forced_text: str,
    max_forced_tokens: int,
) -> tuple[str, list[tuple[str, float]]]:
    """Splice `forced_text` at `cot_position` in the example's CoT, ask the
    model what comes next, return (next_token_str, top-10 (token, logprob)).

    Uses Tinker's `topk_prompt_logprobs` with an appended placeholder token:
    because `topk_prompt_logprobs[i]` is the distribution the model assigns
    to position i conditional on 0..i-1, placing a throwaway token at the
    very end lets us read its slot's top-K as the distribution over the
    **next** token after the splice — one sample() call, exact.
    """
    try:
        import tinker
    except ImportError as e:
        raise BackendNotConfigured(
            f"tinker SDK not importable ({e}); install with `uv pip install tinker`."
        )

    scaffold_root = Path(env["SCAFFOLD_ROOT"])
    meta, source_root, split_of = load_task_meta(scaffold_root, env["AGENT_TASK"])

    # Tinker only serves Qwen-family base models in this scaffold. Rollouts
    # produced by e.g. gemma-3-27b cannot be forced — the next-token
    # distribution from a different model family would not reflect what the
    # actual rolling model would have done.
    source_model = str(meta.get("model", "")).lower()
    if not source_model.startswith("qwen"):
        raise BackendNotConfigured(
            f"`force` is disabled for task {env['AGENT_TASK']!r} because its "
            f"rollouts were generated by model {source_model!r}, which Tinker "
            f"does not serve. Only Qwen-family tasks can be forced in this scaffold."
        )

    sampling_client, tokenizer = _get_sampling_client()
    split = _split_for_agent(env, split_of)
    prefix_ids, cot_ids = build_prompt_parts(
        env["AGENT_TASK"], example, tokenizer, source_root, split
    )

    if cot_position < 0 or cot_position > len(cot_ids):
        raise BackendNotConfigured(
            f"cot_position {cot_position} out of range [0, {len(cot_ids)}] "
            f"for example {example_id} (CoT is {len(cot_ids)} tokens)."
        )

    forced_ids = tokenizer.encode(forced_text, add_special_tokens=False)
    if len(forced_ids) == 0:
        raise BackendNotConfigured(
            f"tokens_to_force {forced_text!r} tokenized to 0 tokens."
        )
    if len(forced_ids) > max_forced_tokens:
        raise BackendNotConfigured(
            f"tokens_to_force is {len(forced_ids)} tokens, limit is {max_forced_tokens}."
        )

    # Any valid vocab id works for the placeholder — the distribution at its
    # slot is conditioned only on the prefix tokens, not on the placeholder
    # itself. Use 0; on every mainstream tokenizer that's a valid id.
    placeholder_id = 0
    prompt_ids = (
        list(prefix_ids)
        + list(cot_ids[:cot_position])
        + list(forced_ids)
        + [placeholder_id]
    )

    # `include_prompt_logprobs=True` is required — Tinker wires it to
    # `prompt_logprobs` on the wire, and the server only fills in
    # `topk_prompt_logprobs` when prompt logprobs are requested.
    future = sampling_client.sample(
        prompt=tinker.ModelInput(
            chunks=[tinker.EncodedTextChunk(tokens=prompt_ids)]
        ),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1, temperature=0.0),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=10,
    )
    response = future.result()

    topk = response.topk_prompt_logprobs
    if not topk or topk[-1] is None:
        raise BackendNotConfigured(
            "Tinker response missing topk_prompt_logprobs at the spliced slot; "
            "the server may not support top-K logprobs for this base model."
        )
    entries = topk[-1]  # list[tuple[token_id, logprob]]

    decoded = [(tokenizer.decode([int(tid)]), float(lp)) for tid, lp in entries]
    decoded.sort(key=lambda x: -x[1])  # defensive; Tinker already orders by logprob
    next_token = decoded[0][0]
    return next_token, decoded[:10]


# ---------------------------------------------------------------------------
# Stubs retained for `ask` — which actually bypasses this backend and goes
# to OpenRouter directly in ask.py, so the stubs here are vestigial.
# ---------------------------------------------------------------------------

def tokenize_count(text: str) -> int:
    return len(text.split())
