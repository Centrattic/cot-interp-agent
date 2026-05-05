# Prompt-family seed for OpenEvolve.
# Modify the instruction strings below. Keep them concise and reusable.
# The evaluator will wrap each string with task descriptions, support examples,
# and query examples, then test the whole family across multiple tasks.

# EVOLVE-BLOCK-START
PROMPT_FAMILY = [
    "Infer the label from stable structural cues in the trace, not from superficial topic words.",
    "Compare the query against labeled examples; identify the most diagnostic differences before deciding.",
    "Act as a monitor: detect signs the model is about to perform the target behavior vs. normal continuation.",
    "Apply a conservative decision rule: predict `1` only when evidence clearly matches the positive-class pattern.",
    "Use a contrastive approach: internally explain why this is *not* the opposite class, then output the final label."
]

# Select a focused subset of shot counts to balance coverage and complexity.
SHOT_COUNTS = [10, 20, 30]
# EVOLVE-BLOCK-END
