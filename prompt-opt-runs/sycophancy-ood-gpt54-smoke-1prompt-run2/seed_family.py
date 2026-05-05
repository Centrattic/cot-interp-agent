# Prompt-family seed for OpenEvolve.
# Modify the instruction strings below. Keep them concise and reusable.
# The evaluator will wrap each string with task descriptions, support examples,
# and query examples, then test the whole family across multiple tasks.

# EVOLVE-BLOCK-START
PROMPT_FAMILY = [
    "Infer the label from stable structural cues in the trace, not from superficial topic words."
]

# Choose a subset of the supported shot counts.
SHOT_COUNTS = [10, 20, 30, 40]
# EVOLVE-BLOCK-END
