#!/usr/bin/env python3
"""Drop-in `codex` replacement that rotates accounts on usage-limit errors.

Reads stdin (the prompt) once, then runs the real codex binary with the same
argv. If the output indicates a rate-limit / forbidden / reconnect error and
more profiles are available, copies the next profile into $CODEX_HOME (or
~/.codex) and retries with the same argv + the same buffered stdin.

Profiles live in /root/.codex-profiles/<name>/{auth.json, config.toml, ...}.
Each profile is a fully-seeded codex home dir (the existing /root/bin/codex-rotate
script uses the same layout).

Activate via either:
  - CODEX_BIN=/path/to/codex_rotating.py in env
  - replace /usr/local/bin/codex with a symlink to this script
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROFILES_DIR = Path(os.environ.get("CODEX_PROFILES_DIR", "/root/.codex-profiles"))
REAL_CODEX = os.environ.get("CODEX_REAL_BIN", "/usr/local/bin/codex.real")
RETRY_PATTERNS = (
    "usage limit",
    "you've hit your usage",
    "403 forbidden",
    "reconnecting",
    "rate limit",
    "rate-limit",
    "quota exceeded",
)
SEED_FILES = ("auth.json", "config.toml", "installation_id", "version.json")


def _codex_home() -> Path:
    raw = os.environ.get("CODEX_HOME", "").strip()
    if raw:
        return Path(raw).expanduser()
    home = os.environ.get("HOME", "")
    return Path(home or "~").expanduser() / ".codex"


def _list_profiles() -> list[Path]:
    if not PROFILES_DIR.exists():
        return []
    profs = sorted(p for p in PROFILES_DIR.iterdir() if p.is_dir() and (p / "auth.json").exists())
    random.shuffle(profs)
    return profs


def _install_profile(prof: Path, codex_home: Path) -> None:
    codex_home.mkdir(parents=True, exist_ok=True)
    for name in SEED_FILES:
        src = prof / name
        if src.exists():
            shutil.copy2(src, codex_home / name)


def _is_rate_limited(blob: str) -> bool:
    low = blob.lower()
    return any(p in low for p in RETRY_PATTERNS)


def _run_once(argv: list[str], stdin_data: str | None) -> tuple[int, str, str]:
    proc = subprocess.run(
        [REAL_CODEX] + argv,
        input=stdin_data,
        text=True,
        capture_output=True,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def main() -> int:
    argv = sys.argv[1:]
    stdin_data = None if sys.stdin.isatty() else sys.stdin.read()

    profiles = _list_profiles()
    home = _codex_home()

    # First attempt with whatever auth is currently installed (no swap),
    # then rotate through other profiles on rate-limit failures.
    attempts: list[Path | None] = [None]
    for p in profiles:
        attempts.append(p)

    last_rc = 99
    last_out = ""
    last_err = ""
    for i, prof in enumerate(attempts):
        if prof is not None:
            try:
                _install_profile(prof, home)
                print(f"[codex_rotating] swapped to profile={prof.name} home={home}", file=sys.stderr)
            except Exception as e:
                print(f"[codex_rotating] failed to install {prof}: {e}", file=sys.stderr)
                continue

        rc, out, err = _run_once(argv, stdin_data)
        last_rc, last_out, last_err = rc, out, err
        combined = out + "\n" + err

        if rc == 0 and not _is_rate_limited(combined):
            sys.stdout.write(out)
            sys.stderr.write(err)
            return rc

        if _is_rate_limited(combined) and i + 1 < len(attempts):
            print(f"[codex_rotating] rate-limit detected, trying next profile", file=sys.stderr)
            time.sleep(1)
            continue

        # Non-rate-limit failure or out of profiles — surface the result.
        sys.stdout.write(out)
        sys.stderr.write(err)
        return rc

    sys.stdout.write(last_out)
    sys.stderr.write(last_err)
    return last_rc


if __name__ == "__main__":
    sys.exit(main())
