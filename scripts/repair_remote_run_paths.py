#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def rewrite_bashrc(path: Path, repo_root: Path) -> None:
    part_dir = path.parent
    run_id = part_dir.parent.name.replace("run-", "")
    task = part_dir.parent.parent.name
    lines = [
        "# Auto-generated bash environment for agent run",
        f'export SCAFFOLD_ROOT="{repo_root.as_posix()}"',
        f'export AGENT_RUN_DIR="{part_dir.as_posix()}"',
        f'export AGENT_TASK="{task}"',
        f'export AGENT_RUN_ID="{run_id}"',
        'export AGENT_BACKEND="codex"',
        f'export PATH="/usr/local/bin:{(repo_root / "bin").as_posix()}:$PATH"',
        'export PYTHON="/usr/bin/python3"',
    ]

    # Preserve partition-specific exports from the existing file.
    existing = path.read_text(encoding="utf-8").splitlines()
    keep_prefixes = (
        "export AGENT_PARTITION_INDEX=",
        "export AGENT_N_PARTITIONS=",
        "export AGENT_DATA_TASK=",
    )
    for line in existing:
        if line.startswith(keep_prefixes):
            lines.append(line)

    codex_home = part_dir / ".codex-home"
    codex_home.mkdir(parents=True, exist_ok=True)
    lines.append(f'export CODEX_HOME="{codex_home.as_posix()}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rewrite_run_tests(path: Path, repo_root: Path) -> None:
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'exec "{(repo_root / "bin" / "run-tests").as_posix()}" "$@"\n',
        encoding="utf-8",
    )
    path.chmod(0o755)


def rewrite_repo_run_tests(repo_root: Path) -> None:
    path = repo_root / "bin" / "run-tests"
    path.write_text(
        "#!/usr/bin/env bash\n"
        "# Test trigger — called by the strategy agent when it runs `test`.\n"
        "# Delegates to src/run_tests.py which launches parallel test agents.\n\n"
        "set -euo pipefail\n\n"
        "if [[ -z \"${SCAFFOLD_ROOT:-}\" ]]; then\n"
        "    echo \"Error: SCAFFOLD_ROOT not set. This command must be run inside an agent session.\"\n"
        "    exit 1\n"
        "fi\n\n"
        'exec "/usr/bin/python3" "$SCAFFOLD_ROOT/src/run_tests.py" "$@"\n',
        encoding="utf-8",
    )
    path.chmod(0o755)


def repair_run(run_dir: Path, repo_root: Path) -> None:
    rewrite_repo_run_tests(repo_root)
    for part in sorted(run_dir.glob("partition-*")):
        bashrc = part / "agent.bashrc"
        if bashrc.exists():
            rewrite_bashrc(bashrc, repo_root)
        wrapper = part / "strategy" / "run-tests"
        if wrapper.exists():
            rewrite_run_tests(wrapper, repo_root)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("run_dirs", nargs="+")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    for run_dir_str in args.run_dirs:
        repair_run(Path(run_dir_str).resolve(), repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
