"""Check the deliberately small Docker context without starting Docker."""

import fnmatch
import shlex
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def matches_path(path, pattern):
    """Root-relative POSIX subset used here; * never consumes a slash.

    Moby patternmatcher.go: Pattern.match/compile and MatchesOrParentMatches.
    https://github.com/moby/patternmatcher/blob/main/patternmatcher.go
    """
    assert "\\" not in pattern
    assert all("**" not in part or part == "**" for part in pattern.split("/"))
    if pattern == "**":
        return True
    if pattern.endswith("/**"):
        return path.startswith(pattern[:-2])
    parts = path.split("/")
    if pattern.startswith("**/"):
        return any(matches_path("/".join(parts[i:]), pattern[3:]) for i in range(len(parts)))
    pattern_parts = pattern.split("/")
    assert "**" not in pattern_parts
    return len(parts) == len(pattern_parts) and all(
        fnmatch.fnmatchcase(part, glob)
        for part, glob in zip(parts, pattern_parts, strict=True)
    )


def included(path):
    """Apply ordered rules to the root-relative path and each ancestor."""
    allowed = True
    for line in (ROOT / ".dockerignore").read_text().splitlines():
        pattern = line.strip()
        if not pattern or pattern.startswith("#"):
            continue
        negative = pattern.startswith("!")
        pattern = pattern.removeprefix("!").rstrip("/")
        parts = path.split("/")
        matches = any(matches_path("/".join(parts[:i]), pattern) for i in range(1, len(parts) + 1))
        if matches:
            allowed = negative
    return allowed


@pytest.mark.parametrize("path", [
    "README.md", "LICENSE", "pyproject.toml", "poetry.lock",
    "requirements/runtime.lock", "requirements/core.lock",
    "quantcortex/__init__.py", "quantcortex/execution/brokers/ib_broker.py",
    "scripts/paper_trade_cycle.py", "scripts/verify_brokers.py",
])
def test_runtime_installation_inputs_remain_in_context(path):
    assert (ROOT / path).is_file()
    assert included(path)


@pytest.mark.parametrize("path", [
    "PRIVATE_REVIEW_SENTINEL.md", "private-paper.pdf", ".env", ".env.production",
    "QuantCortex-AI_flagged.pdf", "Measuring non-US. equity portfolio performance.pdf",
    "AISLOP_CODE_REVIEW.md",
    "reports/private.json", "local_data/prices.csv", "submission_v2/paper.pdf",
    "research/private.ipynb", "quantcortex/__pycache__/base.pyc",
    "quantcortex/.env", "scripts/.env", "scripts/private-review.pdf",
    "requirements/private.txt", "scripts/private/analysis.py",
    "quantcortex/execution/.env.production",
    "quantcortex/execution/__pycache__/base.pyc",
    "quantcortex/execution/private.qc_state.json",
    "quantcortex/execution/.qc_state.json",
    "quantcortex/execution/.venv/bin/python",
    "quantcortex/execution/raw.parquet",
])
def test_private_artifacts_and_caches_stay_out_of_context(path):
    assert not included(path)


def test_copy_uses_only_explicit_runtime_sources():
    sources = []
    for line in (ROOT / "Dockerfile").read_text().splitlines():
        if not line.startswith("COPY "):
            continue
        fields = shlex.split(line)
        sources.extend(field for field in fields[1:-1] if not field.startswith("--"))
    assert set(sources) == {
        "requirements/runtime.lock", "README.md", "LICENSE", "pyproject.toml",
        "poetry.lock", "requirements/", "quantcortex/", "scripts/",
    }


def test_pattern_semantics_are_root_relative_and_recursive_only_when_explicit():
    assert matches_path(".env", ".env")
    assert not matches_path("quantcortex/.env", ".env")
    assert matches_path("quantcortex/execution/.env", "**/.env")
    assert matches_path(".env", "**/.env")
    assert not matches_path("scripts/private/test.py", "scripts/*.py")
    assert matches_path("quantcortex/execution/base.py", "quantcortex/**")
    assert not matches_path("quantcortex", "quantcortex/**")
