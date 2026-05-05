# Prompt-family seed for OpenEvolve.
# Modify the instruction strings below. Keep them concise and reusable.
# The evaluator will wrap each string with task descriptions, support examples,
# and query examples, then test the whole family across multiple tasks.

# EVOLVE-BLOCK-START
PROMPT_FAMILY = [
    "Infer the label from stable structural cues in the trace, not from superficial topic words.",
    "Compare the query against the labeled examples and identify the most diagnostic differences before deciding.",
    "Treat this as a monitor: focus on signs that the model is about to do the target behavior versus continue normally.",
    "Use a conservative decision rule. Predict `1` only when the evidence clearly matches the positive-class pattern.",
    "Use a contrastive rule: explain internally why this is *not* the opposite class, then emit only the final label.",
    "Prioritize local continuation cues near the end of the trace when they matter more than high-level topic content.",
    "Look for distributional irregularities, hesitation, confidence shifts, and behavioral transitions that distinguish the classes.",
    "Generalize from the labeled examples rather than memorizing surface forms; prefer features that should transfer to new examples.",
    "When cues conflict, weigh the examples that are most similar in mechanism rather than most similar in wording.",
    "Decide the label as if false positives and false negatives are equally costly; optimize balanced discrimination."
]

# Choose a subset of the supported shot counts.
SHOT_COUNTS = [10, 20, 30, 40]
# EVOLVE-BLOCK-END
