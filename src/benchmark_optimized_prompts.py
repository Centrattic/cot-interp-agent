from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "optimized-monitor-benchmarks"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.prompt_opt as po

TASKS: dict[str, dict[str, Any]] = {
    "reasoning_termination": {
        "data_task": "reasoning_termination",
        "program": "/home/riya/Riya/research/cot-proxy-tasks/prompt-opt-runs/20260506-reasoning-termination-ood-codex55-q100-w15-fs10-hillclimb-r10/hillclimb_iterations/iteration_00_seed.py",
        "shots": [10],
    },
    "gemma_self_deletion": {
        "data_task": "gemma_self_deletion_clean",
        "program": "/home/riya/Riya/research/cot-proxy-tasks/prompt-opt-runs/20260506-gemma-self-deletion-ood-codex55-q100-w15-fs10-hillclimb-r10/hillclimb_iterations/iteration_02/candidate.py",
        "shots": [10],
    },
    "followup_confidence": {
        "data_task": "followup_confidence",
        "program": "/home/riya/Riya/research/cot-proxy-tasks/prompt-opt-runs/20260506-followup-confidence-ood-codex55-q100-w15-fs30-hillclimb-r10/hillclimb_iterations/iteration_01/candidate.py",
        "shots": [30],
    },
    "user_preference_sycophancy": {
        "data_task": "user_preference_sycophancy",
        "program": "/home/riya/Riya/research/cot-proxy-tasks/prompt-opt-runs/20260506-user-preference-sycophancy-ood-codex55-q100-w15-fs30-hillclimb-r10/hillclimb_iterations/iteration_04/candidate.py",
        "shots": [30],
    },
    "stanford_hint": {
        "data_task": "stanford_hint_clean",
        "program": "/home/riya/Riya/research/cot-proxy-tasks/prompt-opt-runs/20260506-stanford-hint-ood-codex55-q100-w15-fs10-hillclimb-r10/hillclimb_iterations/iteration_00_seed.py",
        "shots": [10],
    },
    "atypical_answer": {
        "data_task": "atypical_answer",
        "program": "/home/riya/Riya/research/cot-proxy-tasks/prompt-opt-runs/20260506-atypical-answer-ood-codex55-q100-w15-fs10-hillclimb-r10/hillclimb_iterations/iteration_00_seed.py",
        "shots": [10],
    },
    "atypical_cot_length": {
        "data_task": "atypical_cot_length",
        "program": "/home/riya/Riya/research/cot-proxy-tasks/prompt-opt-runs/20260506-atypical-cot-length-ood-codex55-q100-w15-fs10-hillclimb-r10/hillclimb_iterations/iteration_04/candidate.py",
        "shots": [10],
    },
}


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": statistics.mean(values), "std": statistics.stdev(values)}


def _run_task(task_key: str, max_parallel_requests: int) -> dict[str, Any]:
    spec = TASKS[task_key]
    out_dir = OUT_ROOT / task_key
    out_dir.mkdir(parents=True, exist_ok=True)

    po.PROJECT_ROOT = ROOT
    po.DATA_DIR = ROOT / "data"

    results: dict[str, Any] = {
        "task": task_key,
        "program": spec["program"],
        "shots": spec["shots"][0],
        "splits": {},
    }

    for split, repeats in (("id_test", 3), ("test", 5)):
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        run_summaries: list[dict[str, Any]] = []

        for run_idx in range(repeats):
            settings = po.OptimizerSettings(
                tasks=[spec["data_task"]],
                random_seed=run_idx,
                query_size=10_000,
                pilot_query_size=10_000,
                top_k_pairs=1,
                episodes=1,
                max_parallel_requests=max_parallel_requests,
                request_timeout_sec=120,
                monitor_backend="codex",
                monitor_model="gpt-5.5",
                monitor_reasoning_effort="medium",
                support_splits=["few-shot"],
                query_splits=[split],
                allowed_shot_counts=list(spec["shots"]),
                prompt_count_schedule=[1],
            )
            task = po.load_task_spec(spec["data_task"], settings)
            support_pool = po._load_records_for_splits(task, ["few-shot"])
            query_pool = po._load_records_for_splits(task, [split])
            prompts, shots = po._load_candidate_module(spec["program"])
            summary, _rows = po._evaluate_prompt_on_queries(
                task=task,
                prompt_index=0,
                prompt_instruction=prompts[0],
                shot_count=shots[0],
                support_pool=support_pool,
                query_examples=query_pool,
                settings=settings,
                support_seed_base=settings.random_seed
                ^ po._stable_hash(spec["program"])
                ^ po._stable_hash(split),
            )
            per_run = {"run_index": run_idx, "split": split, "summary": summary}
            (split_dir / f"run_{run_idx:02d}_summary.json").write_text(json.dumps(per_run, indent=2))
            run_summaries.append(summary)

        agg: dict[str, Any] = {"runs": len(run_summaries)}
        if run_summaries:
            agg["query_size"] = run_summaries[0]["query_size"]
        for key in ("gmean2", "acc", "tpr", "tnr"):
            agg[key] = _summarize([float(row[key]) for row in run_summaries])
        (split_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
        results["splits"][split] = agg

    (out_dir / "result.json").write_text(json.dumps(results, indent=2))
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--max-parallel-requests", type=int, default=15)
    args = parser.parse_args()
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    all_results = [_run_task(task, args.max_parallel_requests) for task in tasks]
    print(json.dumps(all_results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
