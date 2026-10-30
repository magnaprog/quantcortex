"""Report provenance reflects the local input's source-repository Git state."""

import json
import subprocess
from collections import defaultdict
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import generate_report


def test_report_fingerprint_does_not_depend_on_manuscript_tools(tmp_path):
    files = {
        "scripts/generate_report.py": "# report generator\n",
        "pyproject.toml": "[project]\n",
        "poetry.lock": "# dependency lock\n",
        "quantcortex/__init__.py": "",
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="ascii")
    manifest = generate_report.source_tree_manifest(tmp_path)
    assert set(manifest["files"]) == set(files)
    assert manifest["file_count"] == len(files)


@pytest.fixture
def report_repository(tmp_path, monkeypatch):
    repo = tmp_path / "repository"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    script = repo / "scripts" / "generate_report.py"
    script.parent.mkdir()
    script.write_text("# Test-only report source fingerprint.\n", encoding="ascii")
    monkeypatch.setattr(generate_report, "__file__", str(script))
    monkeypatch.setattr(generate_report, "_installed_version", lambda _name: None)
    return repo


def write_manifest(source, output):
    dates = pd.date_range("2024-01-01", periods=2)
    returns = pd.Series(0.0, index=dates)
    metrics = defaultdict(float, dsr_n_trials=1, dsr_sr_variance=None)
    report = defaultdict(
        float,
        m=metrics,
        benchmark_metrics=defaultdict(lambda: {"cagr": 0.0}),
        strategy_configuration={},
        rets=returns,
        cost_model=SimpleNamespace(commission=0.0, slippage=0.0, tax=0.0),
        cash_returns=returns.rename("zero-return cash"),
    )
    generate_report.write_performance_manifest(
        report,
        source,
        pd.DataFrame({"AAA": [100.0, 100.0]}, index=dates),
        output,
        [],
        git_metadata={"source_commit": "test-only", "worktree_clean_at_start": False},
        source_tree={"files": {}, "file_count": 0, "sha256": "0" * 64},
        generated_at="2026-10-10T00:00:00Z",
    )
    return json.loads(output.read_text(encoding="utf-8"))


@pytest.mark.parametrize("state", ["tracked", "untracked", "ignored", "outside"])
def test_local_manifest_records_git_tracking(report_repository, tmp_path, state):
    repo = report_repository
    source = (tmp_path if state == "outside" else repo) / "prices.csv"
    source.write_text("date,AAA\n2024-01-01,100\n", encoding="ascii")
    if state == "tracked":
        subprocess.run(["git", "add", "--", source.name], cwd=repo, check=True)
    elif state == "ignored":
        (repo / ".gitignore").write_text("prices.csv\n", encoding="ascii")

    manifest = write_manifest(
        {"kind": "local CSV", "path": str(source), "sha256": "a" * 64},
        tmp_path / "output" / "manifest.json",
    )

    assert manifest["source"]["raw_input_committed"] is (state == "tracked")
    assert manifest["source"]["input_sha256"] == "a" * 64


@pytest.mark.parametrize("track_literal", [False, True])
def test_manifest_matches_literal_input_path(report_repository, tmp_path, track_literal):
    repo = report_repository
    literal = repo / "prices[1].csv"
    wildcard_match = repo / "prices1.csv"
    for path in (literal, wildcard_match):
        path.write_text("date,AAA\n2024-01-01,100\n", encoding="ascii")
    tracked = literal if track_literal else wildcard_match
    subprocess.run(
        ["git", "--literal-pathspecs", "add", "--", tracked.name], cwd=repo, check=True
    )

    manifest = write_manifest(
        {"kind": "local CSV", "path": str(literal)},
        tmp_path / "output" / "manifest.json",
    )

    assert manifest["source"]["raw_input_committed"] is track_literal


@pytest.mark.parametrize("path", [None, ""])
def test_local_manifest_requires_path_before_writing(report_repository, tmp_path, path):
    source = {"kind": "local CSV"}
    if path is not None:
        source["path"] = path
    output = tmp_path / "output" / "manifest.json"

    with pytest.raises(ValueError, match="local CSV.*path"):
        write_manifest(source, output)

    assert not output.parent.exists()


@pytest.mark.parametrize("failure", ["git-error", "unavailable"])
def test_tracking_errors_fail_before_writing(
    report_repository, tmp_path, monkeypatch, failure
):
    source = report_repository / "prices.csv"
    source.write_text("date,AAA\n2024-01-01,100\n", encoding="ascii")
    output = tmp_path / "output" / "manifest.json"

    def fail_git(*args, **kwargs):
        if failure == "unavailable":
            raise OSError("Git unavailable")
        return SimpleNamespace(returncode=128, stderr="fatal: cannot read index")

    monkeypatch.setattr(generate_report.subprocess, "run", fail_git)
    with pytest.raises(RuntimeError, match="could not determine.*tracked"):
        write_manifest({"kind": "local CSV", "path": str(source)}, output)

    assert not output.parent.exists()


def test_live_manifest_has_no_local_tracked_input(report_repository, tmp_path, monkeypatch):
    def unexpected_git(*args, **kwargs):
        pytest.fail("a live source without a local input does not need a Git path check")

    monkeypatch.setattr(generate_report.subprocess, "run", unexpected_git)
    manifest = write_manifest(
        {"kind": "live yfinance", "provider": "test-only provider metadata"},
        tmp_path / "output" / "manifest.json",
    )

    assert manifest["source"]["raw_input_committed"] is False
