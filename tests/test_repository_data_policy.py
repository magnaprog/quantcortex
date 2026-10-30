from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_MARKET_DATA = [
    "data/sample/rotation_prices.csv",
    "quantcortex/data/sample/rotation_prices.csv",
]
RAW_MARKET_DATA_SIGNATURES = (
    {"date", "QQQ", "VGT", "GLD", "TLT", "SPY", "VIG", "SHV"},
    {"date", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "SHV"},
    {"date", "EWA", "EWC", "EWG", "EWH", "EWJ", "EWL", "EWP", "EWQ", "EWS", "EWU", "SHV"},
)
PUBLISHED_CHARTS = {
    "allocation_and_exposure.png",
    "drawdown.png",
    "equity_vs_benchmarks.png",
    "monthly_returns.png",
    "performance_attribution.png",
    "report_overview.png",
    "return_distribution.png",
    "rolling_risk.png",
    "rolling_sharpe.png",
    "turnover_and_costs.png",
}
DOCUMENTATION_FILES = [
    "README.md",
    "PERFORMANCE.md",
    "CONTRIBUTING.md",
    "AGENTS.md",
    "CLAUDE.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "local_data/README.md",
    "docs/architecture.md",
    "docs/data-source-due-diligence.md",
    "docs/evaluation-contracts.md",
    "docs/onboarding.md",
    "docs/production-readiness.md",
]


def test_redistributed_market_data_is_absent():
    present = [path for path in FORBIDDEN_MARKET_DATA if (REPO_ROOT / path).exists()]
    assert not present, f"redistributed market-data snapshots reappeared: {present}"

    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert [path for path in tracked if path.startswith("local_data/")] == [
        "local_data/README.md"
    ]

    leaked = []
    for relative_path in tracked:
        if not relative_path.endswith(".csv"):
            continue
        with (REPO_ROOT / relative_path).open(encoding="utf-8", newline="") as handle:
            header = set(next(csv.reader(handle), []))
        if any(signature <= header for signature in RAW_MARKET_DATA_SIGNATURES):
            leaked.append(relative_path)
    assert not leaked, f"tracked raw paper price matrices detected: {leaked}"


def test_published_performance_charts_match_manifest_and_readme():
    image_dir = REPO_ROOT / "docs" / "img"
    manifest_path = image_dir / "performance_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 4

    source = manifest["source"]
    assert source["permission_basis"]
    assert source["raw_input_committed"] is False
    input_digest = source["input_sha256"]
    assert len(input_digest) == 64
    int(input_digest, 16)

    artifacts = manifest["artifacts"]
    assert set(artifacts) == PUBLISHED_CHARTS
    assert {path.name for path in image_dir.glob("*.png")} == PUBLISHED_CHARTS
    for name, expected_digest in artifacts.items():
        artifact = image_dir / name
        actual_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert actual_digest == expected_digest, name

    source_tree = manifest["generator"]["source_tree"]
    expected_sources = {
        "scripts/generate_report.py", "pyproject.toml", "poetry.lock",
    } | {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "quantcortex").rglob("*.py")
    }
    assert set(source_tree["files"]) == expected_sources
    assert source_tree["file_count"] == len(source_tree["files"])
    aggregate_digest = hashlib.sha256()
    for relative, expected_digest in source_tree["files"].items():
        source_path = REPO_ROOT / relative
        actual_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        assert actual_digest == expected_digest, relative
        aggregate_digest.update(relative.encode("utf-8"))
        aggregate_digest.update(b"\0")
        aggregate_digest.update(bytes.fromhex(expected_digest))
    assert aggregate_digest.hexdigest() == source_tree["sha256"]

    git_metadata = manifest["generator"]["git"]
    source_commit = git_metadata["source_commit"]
    assert len(source_commit) == 40
    int(source_commit, 16)
    assert git_metadata["worktree_clean_at_start"] is True

    reproduction = (REPO_ROOT / "docs/data-source-due-diligence.md").read_text(
        encoding="utf-8"
    )
    assert input_digest in reproduction
    assert "img/performance_manifest.json" in reproduction
    performance = (REPO_ROOT / "PERFORMANCE.md").read_text(encoding="utf-8")
    for name in PUBLISHED_CHARTS:
        assert f"docs/img/{name}" in performance
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "PERFORMANCE.md" in readme
    assert "docs/data-source-due-diligence.md" in readme


def test_research_notebooks_have_no_committed_execution_state():
    for path in sorted((REPO_ROOT / "research").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            assert cell.get("execution_count") is None, f"{path}:{index} is executed"
            assert not cell.get("outputs"), f"{path}:{index} has committed outputs"
            assert "execution" not in cell.get("metadata", {}), (
                f"{path}:{index} has execution timestamps"
            )


def test_published_documentation_metrics_match_report_manifest():
    manifest = json.loads(
        (REPO_ROOT / "docs/img/performance_manifest.json").read_text(encoding="utf-8")
    )
    metrics = manifest["metrics"]
    formats = {
        "net_nominal_cagr": "+.2%",
        "gross_nominal_cagr_before_modeled_costs": "+.2%",
        "annualized_volatility": ".2%",
        "net_cash_excess_sharpe": "+.2f",
        "gross_cash_excess_sharpe_before_modeled_costs": "+.2f",
        "max_drawdown": "+.2%",
        "annualized_one_way_turnover": ".2f",
        "annualized_gross_traded_notional": ".2f",
        "arithmetic_sum_transaction_cost_return_drag": ".2%",
        "mean_active_gross_exposure": ".2%",
        "fully_cash_session_fraction": ".2%",
        "cash_proxy_cagr": "+.2%",
        "exposure_matched_equal_weight_cash_excess_sharpe_gross": "+.2f",
    }
    document = (REPO_ROOT / "PERFORMANCE.md").read_text(encoding="utf-8")
    for field, spec in formats.items():
        value = format(metrics[field], spec)
        if field in {"annualized_one_way_turnover", "annualized_gross_traded_notional"}:
            value += "x"
        assert value in document, f"PERFORMANCE.md metric drifted: {field}={value}"


def test_manuscript_and_presentation_assets_are_absent():
    tracked = subprocess.check_output(
        ["git", "ls-files"], cwd=REPO_ROOT, text=True
    ).splitlines()
    forbidden_suffixes = {".pdf", ".tex", ".sty", ".bib", ".pptx", ".drawio", ".zip"}
    unexpected = [
        name for name in tracked
        if name.startswith(("paper/", "poster/", "posters/"))
        or Path(name).suffix in forbidden_suffixes
    ]
    assert not unexpected, f"manuscript or presentation assets in public tree: {unexpected}"


def test_survivorship_demo_does_not_diagnose_missing_price_responses(monkeypatch, capsys):
    from types import SimpleNamespace

    import pandas as pd

    from quantcortex.data.providers.yfinance_provider import YFinanceProvider
    from quantcortex.data.universe.sp500_universe import SP500Universe
    from scripts.survivorship_demo import main

    universe = SimpleNamespace(
        constituents=lambda as_of=None: ["MISSING"] if as_of is not None else []
    )
    monkeypatch.setattr(
        SP500Universe, "from_wikipedia", classmethod(lambda cls: universe)
    )
    monkeypatch.setattr(
        YFinanceProvider, "get_prices", lambda *args, **kwargs: pd.DataFrame()
    )

    assert main(["survivorship_demo.py", "2020-01-01", "--live-yfinance"]) == 0
    output = capsys.readouterr().out
    assert "1 returned no usable prices" in output
    assert "Missing prices do not identify delistings, mergers, or renamings." in output
    assert "are NOT (delisted/" not in output
    assert "examples no longer priceable" not in output


def test_documentation_local_links_resolve():
    missing = []
    for relative_path in DOCUMENTATION_FILES:
        document = REPO_ROOT / relative_path
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"!?\[[^]]*\]\(([^)]+)\)", text):
            target = target.strip().split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{relative_path}: {target}")
    assert not missing, f"broken local documentation links: {missing}"
