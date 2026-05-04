"""Filesystem-level anti-cheat guard for Codex scaffold agents.

Codex CLI's workspace-write sandbox does not prevent read access to arbitrary
absolute paths on this machine. For scaffolded agents, that means prompts alone
are not enough: a strategy/test agent can read held-out labels from
data/<task>/test or raw rollout metadata under data/<task>/qwen-3-32b.

This guard temporarily removes all permissions from those directory roots while
a Codex agent subprocess is running. The parent scaffold restores permissions
before it needs to collect/copy test examples.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
from types import TracebackType


class HeldoutPathGuard:
    def __init__(self, repo_root: Path, task_name: str | None = None):
        self.repo_root = repo_root.resolve()
        self.task_name = task_name
        scope = f"{self.repo_root}:{task_name or 'all'}"
        digest = hashlib.sha1(scope.encode("utf-8")).hexdigest()[:16]
        self.state_path = Path("/tmp") / f"cot_interp_anti_cheat_{digest}.json"
        self.lock_path = Path("/tmp") / f"cot_interp_anti_cheat_{digest}.lock"
        self.lock_file = None

    def _paths(self) -> list[Path]:
        out: list[Path] = []
        data = self.repo_root / "data"
        task_dirs: list[Path] = []
        if data.exists() and self.task_name:
            task_dir = data / self.task_name
            if task_dir.is_dir():
                task_dirs = [task_dir]
        elif data.exists():
            task_dirs = [p for p in data.iterdir() if p.is_dir()]

        for task_dir in task_dirs:
            for child in (task_dir / "test", task_dir / "qwen-3-32b"):
                if child.is_dir():
                    out.append(child.resolve())
            meta_path = task_dir / "metadata.json"
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            except Exception:
                meta = {}
            source = meta.get("source")
            if source:
                source_path = Path(source)
                if source_path.is_dir():
                    out.append(source_path.resolve())

        if self.task_name:
            return sorted(set(out))

        datasets = self.repo_root / "datasets"
        if datasets.exists():
            for dataset_dir in datasets.iterdir():
                if not dataset_dir.is_dir():
                    continue
                for model_dir in dataset_dir.iterdir():
                    if model_dir.is_dir():
                        out.append(model_dir.resolve())
        return sorted(set(out))

    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return {"count": 0, "modes": {}}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {"count": 0, "modes": {}}

    def _save_state(self, state: dict) -> None:
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def __enter__(self) -> "HeldoutPathGuard":
        self.lock_file = self.lock_path.open("a+")
        fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX)
        state = self._load_state()
        if int(state.get("count", 0)) == 0:
            modes = {}
            for path in self._paths():
                try:
                    modes[str(path)] = path.stat().st_mode & 0o777
                    path.chmod(0)
                except FileNotFoundError:
                    continue
            state = {"count": 1, "modes": modes}
        else:
            state["count"] = int(state.get("count", 0)) + 1
        self._save_state(state)
        fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self.lock_file is None:
            return
        fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX)
        state = self._load_state()
        count = max(0, int(state.get("count", 0)) - 1)
        if count == 0:
            for raw_path, mode in state.get("modes", {}).items():
                path = Path(raw_path)
                if path.exists():
                    path.chmod(int(mode))
            if self.state_path.exists():
                self.state_path.unlink()
        else:
            state["count"] = count
            self._save_state(state)
        fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
        self.lock_file.close()
        self.lock_file = None


def restore_heldout_paths(repo_root: Path) -> None:
    """Best-effort manual restoration after an interrupted guarded run."""
    guard = HeldoutPathGuard(repo_root)
    tmp = Path("/tmp")
    candidates = set(tmp.glob("cot_interp_anti_cheat_*.json"))
    candidates.add(guard.state_path)
    for state_path in sorted(candidates):
        digest = state_path.stem.removeprefix("cot_interp_anti_cheat_")
        lock_path = tmp / f"cot_interp_anti_cheat_{digest}.lock"
        lock = lock_path.open("a+")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
            except Exception:
                state = {}
            for raw_path, mode in state.get("modes", {}).items():
                path = Path(raw_path)
                if path.exists():
                    path.chmod(int(mode))
            if state_path.exists():
                state_path.unlink()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
