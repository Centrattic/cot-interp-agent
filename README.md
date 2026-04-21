# cot-interp-agent

Scaffold for running interpretability-research agents on binary classification
tasks derived from `cot-proxy-tasks`. Each task asks an agent to look at one
partial qwen-3-32b rollout and predict some binary property of what the model
does next.

Quick reference:

```bash
./scaffold.sh run <task_name>                    # legacy single-strategy run
./scaffold.sh run <task_name> --n-strategies 10  # 10-way cross-val partitioning
./scaffold.sh status                             # list runs
python src/ingest_cot_proxy.py --preset <task>   # populate data/<task>/
```

See `CLAUDE.md` for the directory layout and tool-authoring conventions.

## Multi-partition runs (`--n-strategies N`)

A single-strategy run (`--n-strategies 1`, the default) trains one strategy
agent on one fixed few-shot sample and evaluates it against the *entire*
`data/<task>/test/` set. That's fine for smoke-testing, but it blends two
sources of variance (strategy quality and few-shot luck) and doesn't stress
generalization — if the sampled few-shot happens to be representative, the
strategy can overfit to it without penalty.

`--n-strategies N` with `N > 1` switches to a cross-val-style layout:

- **N freshly-sampled few-shot sets.** Partition `k` samples its own balanced
  few-shot from the task's source training split using
  `seed = --strategy-seed-base + k` (default base = 0). Each partition sees
  different examples.
- **N strategy agents in parallel.** They develop independent `STRATEGY.md`
  files in `run-<ts>/partition-000/strategy/` … `partition-NNN/strategy/`.
  Parallelism is capped by `AGENT_STRATEGY_PARALLEL` (default 10).
- **Disjoint test slices.** When partition `k`'s strategy calls `test`, it
  only runs on a round-robin slice of the test set: examples at indices
  `k, k+N, k+2N, …`. All N slices together cover the whole test set exactly
  once, so aggregate metrics are computed on full test coverage — just with
  each test example scored by the strategy that *didn't* see it.
- **Aggregate scoring.** After all partitions finish, `score_run.score_partitioned_run`
  writes `run-<ts>/results.csv` (one row per test example, flagged with the
  partition that produced it) plus `run-<ts>/summary.txt`.

Example:

```bash
./scaffold.sh run reasoning_termination --n-strategies 10
# -> run-<ts>/partition-000/{strategy,test-NNN}/
#    run-<ts>/partition-001/{strategy,test-NNN}/
#    ...
#    run-<ts>/results.csv          # merged test results
#    run-<ts>/summary.txt          # overall + per-partition metrics
```

Relevant flags:

| flag | default | effect |
| --- | --- | --- |
| `--n-strategies N` | 1 | number of partitions (and parallel strategy agents) |
| `--strategy-seed-base B` | 0 | partition `k` uses seed `B + k` for its few-shot sample |
| `--few-shot-per-class K` | from metadata | override per-class few-shot size |
| `--tools t1,t2,…` | `` | custom research tools enabled this run (see `scaffold.py:TOOL_DESCRIPTIONS`) |

Env vars worth knowing:
- `AGENT_STRATEGY_PARALLEL` — cap on concurrent strategy agents (default 10).
- `AGENT_TEST_MAX_WORKERS` — cap on concurrent test agents *per partition*
  (default 10). With `N=10` partitions this means up to 100 test agents in
  flight, so bump down on rate-limit-constrained accounts.
