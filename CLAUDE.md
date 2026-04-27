# Agent Scaffold

This is a scaffold for running interpretability research agents on binary classification tasks.

## Structure

- `data/` — Task data (7 independent tasks), each with `few-shot/` and `test/` subdirectories containing `.json` examples and `.npy` activations
- `src/scaffold.py` — Main orchestrator (init, run, status)
- `src/run_tests.py` — Parallel test runner launched by the `test` command
- `src/tools/` — Python backends for custom agent tools (token forcing, probes, etc.)
- `bin/` — Shell wrappers for tools, added to agent PATH
- `agent-runs/` — Per-task, per-run directories with strategy/ and test/ outputs
- `agent-traces/` — Claude session traces for debugging
- `prompts/` — System prompts for strategy and test agents

## Usage

```bash
./scaffold.sh init                    # Create directories
./scaffold.sh run <task_name>         # Launch strategy agent
./scaffold.sh status                  # Show all runs
```

## Adding Custom Tools

1. Add Python implementation in `src/tools/`
2. Create a shell wrapper in `bin/` that calls it
3. Document the command in the generated tool docs used by `strategy/README.md` and prompt assembly
