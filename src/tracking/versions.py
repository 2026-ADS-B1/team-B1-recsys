"""Code and data version of a training, saved as run tags (US3.2, task 3 / AC3).

Code version (Git):
    mlflow.source.git.commit / .branch / .repoURL  -> shown by the MLflow UI ("Version")
    code.git_commit, code.git_branch, code.commit_url
    code.git_dirty = "true" if tracked files had uncommitted changes: then the commit
    alone is not enough to repeat the run.

Data version (DVC):
    data.version.<path> = DVC hash of each versioned dataset used (from its .dvc file
    or from dvc.lock), e.g. data.version.data/raw/ml-1m = 82cb...dir
    data.modified_locally = "false" if the local files match those hashes,
    "true" if they don't, "unknown" if DVC is not installed.

Every value is read, never guessed: when something can't be known it says "unknown".
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

import yaml

UNKNOWN = "unknown"
SKIP_DIRS = {".git", ".dvc", ".venv", "venv", "node_modules", "mlruns", "__pycache__"}


# ---------------------------------------------------------------------------- code
def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=20, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _public_url(remote: str) -> str:
    """https://user:token@github.com/org/repo.git -> https://github.com/org/repo (no secrets)."""
    remote = re.sub(r"^(https?://)[^@/]+@", r"\1", remote.strip())
    remote = re.sub(r"^git@([^:]+):", r"https://\1/", remote)
    return remote[:-4] if remote.endswith(".git") else remote


def code_version(repo_root: str | Path = ".") -> dict[str, str]:
    """Git commit, branch, repository URL and whether there were uncommitted changes."""
    root = Path(repo_root)
    commit = _git(["rev-parse", "HEAD"], root)
    if not commit:
        return {"code.git_commit": UNKNOWN, "code.git_dirty": UNKNOWN}

    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root) or UNKNOWN
    changes = _git(["status", "--porcelain", "--untracked-files=no"], root)
    dirty = UNKNOWN if changes is None else str(bool(changes)).lower()
    remote = _git(["config", "--get", "remote.origin.url"], root)

    tags = {
        "code.git_commit": commit,
        "code.git_branch": branch,
        "code.git_dirty": dirty,
        "mlflow.source.git.commit": commit,
        "mlflow.source.git.branch": branch,
    }
    if remote:
        url = _public_url(remote)
        tags["mlflow.source.git.repoURL"] = url
        if url.startswith("https://github.com/"):
            tags["code.commit_url"] = f"{url}/commit/{commit}"
    return tags


# ---------------------------------------------------------------------------- data
def tracked_data(repo_root: str | Path = ".") -> dict[str, str]:
    """{data path: DVC hash} for every .dvc pointer file and every dvc.lock output."""
    root = Path(repo_root)
    found: dict[str, str] = {}

    for pointer in root.rglob("*.dvc"):
        if any(part in SKIP_DIRS for part in pointer.relative_to(root).parts) or not pointer.is_file():
            continue
        content = yaml.safe_load(pointer.read_text(encoding="utf-8")) or {}
        for out in content.get("outs") or []:
            md5 = out.get("md5") or out.get("hash")
            if out.get("path") and md5:
                path = (pointer.parent / out["path"]).relative_to(root).as_posix()
                found[path] = str(md5)

    lock = root / "dvc.lock"
    if lock.exists():
        content = yaml.safe_load(lock.read_text(encoding="utf-8")) or {}
        for stage in (content.get("stages") or {}).values():
            for out in stage.get("outs") or []:
                md5 = out.get("md5") or out.get("hash")
                if out.get("path") and md5:
                    found[Path(out["path"]).as_posix()] = str(md5)
    return found


def _relative(path: str, root: Path) -> str:
    """Path relative to the repository when possible, with forward slashes."""
    p = Path(path)
    if p.is_absolute():
        try:
            p = p.resolve().relative_to(root.resolve())
        except ValueError:
            pass
    return p.as_posix().strip("/")


def _tag_key(path: str) -> str:
    """MLflow tag names only allow letters, digits, _ - . space and /."""
    return "data.version." + re.sub(r"[^A-Za-z0-9_\-. /]", "_", path)


def _covering(path: str, tracked: Iterable[str]) -> str | None:
    """The tracked path that contains `path` (itself or a parent folder)."""
    path = Path(path).as_posix().strip("/")
    for candidate in sorted(tracked, key=len, reverse=True):
        if path == candidate or path.startswith(candidate + "/"):
            return candidate
    return None


def _modified_locally(targets: list[str], repo_root: Path) -> str:
    """Compare the local files with their DVC hashes (needs the dvc package)."""
    try:
        from dvc.repo import Repo
    except ImportError:
        return UNKNOWN
    try:
        with Repo(str(repo_root)) as repo:
            status = repo.status(targets=targets)
    except Exception:  # not a DVC repo, broken cache...
        return UNKNOWN
    return str(bool(status)).lower()


def data_version(
    data_paths: Iterable[str] | None = None,
    repo_root: str | Path = ".",
    check_local: bool = True,
) -> dict[str, str]:
    """DVC hash of each dataset used (all versioned data if data_paths is None)."""
    root = Path(repo_root)
    tracked = tracked_data(root)
    tags: dict[str, str] = {}

    if data_paths is None:
        used = sorted(tracked)
    else:
        used = []
        for path in data_paths:
            path = _relative(path, root)
            covering = _covering(path, tracked)
            if covering is None:
                tags[_tag_key(path)] = "not versioned with DVC"
            elif covering not in used:
                used.append(covering)

    for path in used:
        tags[_tag_key(path)] = tracked[path]

    if not used:
        tags["data.modified_locally"] = UNKNOWN
    elif check_local:
        pointers = [p for p in (f"{u}.dvc" for u in used) if (root / p).exists()]
        tags["data.modified_locally"] = _modified_locally(pointers or used, root)
    return tags


def version_tags(
    data_paths: Iterable[str] | None = None,
    repo_root: str | Path = ".",
    check_local: bool = True,
) -> dict[str, str]:
    """All the tags of this module: code version + data version."""
    return {
        **code_version(repo_root),
        **data_version(data_paths, repo_root, check_local=check_local),
    }


def warnings_for(tags: dict[str, str]) -> list[str]:
    """Human messages when the run won't be exactly repeatable."""
    messages = []
    if tags.get("code.git_dirty") == "true":
        messages.append(
            "There are uncommitted code changes: commit them first if this run must be repeatable."
        )
    if tags.get("data.modified_locally") == "true":
        messages.append(
            "Your local data differs from its DVC version: run 'dvc pull' (or 'dvc add' + commit)."
        )
    return messages
