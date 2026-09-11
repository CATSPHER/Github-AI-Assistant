"""
V1: just clone the repo with GitPython and read files off disk.
(V3 later replaces/augments this with an MCP GitHub server so the agent can
also pull issues, PRs, and commit metadata live instead of only files.)
"""
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass

import git

from . import config


@dataclass
class RepoFile:
    path: str          # relative path, used as citation / metadata
    content: str


def clone_repo(repo_url: str) -> Path:
    """Clones into a fresh temp dir and returns the local path."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="repo_"))
    git.Repo.clone_from(repo_url, tmp_dir, depth=1)  # depth=1: we only need current files for V1
    return tmp_dir


def load_files(repo_dir: Path) -> list[RepoFile]:
    files = []
    for path in repo_dir.rglob("*"):
        if not path.is_file():
            continue
        if any(part in config.IGNORED_DIRS for part in path.parts):
            continue
        if path.suffix not in config.ALLOWED_EXTENSIONS and path.name not in config.ALLOWED_FILENAMES:
            continue
        if path.stat().st_size > config.MAX_FILE_SIZE_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # binary-ish file that slipped past the extension filter

        rel_path = str(path.relative_to(repo_dir))
        files.append(RepoFile(path=rel_path, content=text))
    return files


def cleanup(repo_dir: Path):
    shutil.rmtree(repo_dir, ignore_errors=True)