from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

import pytest
from PIL import Image

from scripts import publish_artifact_bundle
from scripts.run_paper_experiments import (
    _json_sha256,
    source_tree_manifest,
    verify_expansion_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_ROOT = REPO_ROOT / "paper"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protocol_figure_uses_reviewed_drawio_exports():
    with Image.open(PAPER_ROOT / "audit_protocol.drawio.png") as image:
        assert unquote(image.info["mxGraphModel"]) == (
            PAPER_ROOT / "audit_protocol.drawio"
        ).read_text()
    manifest = json.loads((PAPER_ROOT / "results" / "manifest.json").read_text())
    sources = manifest["generator"]["source_tree"]["files"]
    for name in (
        "audit_protocol.drawio",
        "audit_protocol.drawio.pdf",
        "audit_protocol.drawio.png",
    ):
        assert sources[f"paper/{name}"] == _sha256(PAPER_ROOT / name)
    for suffix in ("pdf", "png"):
        assert (PAPER_ROOT / "figures" / f"audit_protocol.{suffix}").read_bytes() == (
            PAPER_ROOT / f"audit_protocol.drawio.{suffix}"
        ).read_bytes()


def test_paper_artifacts_match_manifest_and_generator():
    manifest = json.loads(
        (PAPER_ROOT / "results" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 5

    source = manifest["source"]
    assert source["raw_input_committed"] is False
    assert source["permission_basis"]
    assert len(source["input_sha256"]) == 64
    int(source["input_sha256"], 16)

    generator = manifest["generator"]
    script = REPO_ROOT / generator["path"]
    assert script.is_file()
    assert _sha256(script) == generator["script_sha256"]
    source_commit = generator["git"]["source_commit"]
    assert len(source_commit) == 40
    int(source_commit, 16)
    assert generator["git"]["worktree_clean_at_start"] is True
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            source_commit,
            "--",
            "quantcortex",
            "scripts/publish_artifact_bundle.py",
            "scripts/release_paper_artifacts.sh",
            "scripts/run_paper_experiments.py",
            "schemas",
            "pyproject.toml",
            "poetry.lock",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    dependency_lock = generator["dependency_lock"]
    assert dependency_lock["path"] == "poetry.lock"
    assert dependency_lock["sha256"] == _sha256(REPO_ROOT / "poetry.lock")
    assert generator["threadpools"]
    source_tree = generator["source_tree"]
    assert "scripts/publish_artifact_bundle.py" in source_tree["files"]
    assert "scripts/release_paper_artifacts.sh" in source_tree["files"]
    assert source_tree == source_tree_manifest(
        REPO_ROOT,
        list(source_tree["files"]),
    )

    design = manifest["design"]
    assert design["strategy_parameters"] == {
        "ir_lookback": 126,
        "max_position_weight": 0.6,
        "mom_gap": 21,
        "mom_lookback": 126,
        "regime_backend": "gmm",
        "regime_covariance_type": "full",
        "regime_feature_vol_lookback": 20,
        "regime_n_iter": 100,
        "regime_n_states": 3,
        "regime_reg_covar": 1e-05,
        "regime_seed": 42,
        "target_vix": 20.0,
        "top_n_groups": 2,
        "vix_cap": 1.0,
        "vix_floor": 0.3,
        "vix_proxy_lookback": 21,
    }
    assert design["bootstrap"]["sensitivity_block_lengths_sessions"] == [5, 21, 63]
    assert design["bootstrap"]["ablation_intervals"].startswith("joint across")
    assert design["input_validation"] == {
        "complete_rows_required": True,
        "max_forward_fill_sessions": 0,
    }
    assert design["warmup"] == {
        "available_sessions": 503,
        "required_sessions": 274,
    }
    assert design["signal_fallback"].startswith(
        "hold cash when mature selected-group residual scores"
    )
    assert design["return_decomposition"]["method"] == (
        "exact daily arithmetic identity"
    )
    assert design["bootstrap"]["interval"] == (
        "two-sided unstudentized percentile interval"
    )
    assert design["primary_engine"] == "event_driven"
    assert "pseudo-share quantities stay fixed" in design["primary_engine_semantics"]
    assert "portfolio weights drift" in design["primary_engine_semantics"]
    assert manifest["configuration_sha256"] == _json_sha256(design)
    assert design["universe"]["group_display_names"] == {
        "growth": "growth equities",
        "real_assets": "gold and nominal duration",
        "defensive": "broad and dividend equities",
    }
    assert design["comparators"]["realized_exposure_attribution_control"].endswith(
        "ex-post and gross of comparator costs"
    )
    assert "next-bar execution" in design["comparators"][
        "target_exposure_costed_comparator"
    ]

    contract_metadata = manifest["evaluation_contract"]
    contract_path = PAPER_ROOT / contract_metadata["path"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract_metadata["schema_version"] == 1
    assert contract_metadata["canonical_sha256"] == _json_sha256(contract)
    assert contract["comparators"]["realized_exposure_attribution_control"][
        "implementable"
    ] is False
    assert contract["comparators"]["target_exposure_costed_comparator"][
        "costed"
    ] is True
    assert contract["overlays"]["may_reduce_risky_exposure"] is True
    assert "block automatic retry" in contract["order_state"][
        "uncertain_submission"
    ]

    generated_values = (
        PAPER_ROOT / "results" / "generated_values.tex"
    ).read_text(encoding="ascii")
    assert source["input_sha256"] in generated_values
    assert generator["source_tree"]["sha256"] in generated_values
    assert "\\newcommand{\\PaperWarmupSessions}{503}" in generated_values
    assert "\\newcommand{\\PaperRequiredWarmupSessions}{274}" in generated_values
    assert "\\newcommand{\\PaperBootstrapReplications}{5,000}" in generated_values
    assert "\\newcommand{\\PaperGrossActiveFullLower}" in generated_values
    assert "\\newcommand{\\PaperGrossTradedNotional}" in generated_values
    assert "\\newcommand{\\PaperCostedComparatorCAGR}" in generated_values
    assert "\\newcommand{\\PaperCostedActiveSharpeLower}" in generated_values
    assert "\\newcommand{\\PaperDecompositionRows}" in generated_values

    artifacts = manifest["artifacts"]
    assert artifacts
    assert "results/ablation_uncertainty.csv" in artifacts
    assert "results/comparator_diagnostics.csv" in artifacts
    assert "results/evaluation_contract.json" in artifacts
    assert "results/target_tape_hashes.json" in artifacts
    assert "results/return_decomposition.csv" in artifacts
    assert "results/sharpe_uncertainty.csv" in artifacts
    assert "results/protocol_switches.csv" in artifacts
    assert "figures/bootstrap_robustness.pdf" in artifacts
    assert "figures/return_attribution_and_protocol_switches.pdf" in artifacts
    for relative_path, expected_digest in artifacts.items():
        artifact = PAPER_ROOT / relative_path
        assert artifact.is_file(), relative_path
        assert _sha256(artifact) == expected_digest, relative_path

    target_tape_path = PAPER_ROOT / manifest["decision_streams"]["path"]
    target_tapes = json.loads(target_tape_path.read_text(encoding="utf-8"))
    assert target_tapes == manifest["decision_streams"]["variants"]
    assert set(target_tapes) == {"full", "no_regime", "no_vol_scaler", "signal_only"}
    for metadata in target_tapes.values():
        assert metadata["symbols"] == sorted(metadata["symbols"])
        assert metadata["record_count"] == (
            metadata["decision_count"] * len(metadata["symbols"])
        )
        assert len(metadata["canonical_payload_sha256"]) == 64
        int(metadata["canonical_payload_sha256"], 16)

    with (PAPER_ROOT / "results" / "protocol_switches.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        protocols = {row["protocol"]: row for row in csv.DictReader(handle)}
    assert protocols["audited"]["diagnostic_class"] == "reference"
    assert protocols["zero_return_cash"]["diagnostic_class"] == (
        "economic_counterfactual"
    )
    assert protocols["zero_modeled_costs"]["diagnostic_class"] == (
        "economic_counterfactual"
    )
    assert protocols["invalid_same_close"]["diagnostic_class"] == (
        "causally_invalid"
    )
    assert protocols["invalid_same_close"]["causally_valid"] == "False"

    with (PAPER_ROOT / "results" / "ablation_uncertainty.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        ablation_intervals = list(csv.DictReader(handle))
    assert {row["variant"] for row in ablation_intervals} == {
        "full",
        "no_regime",
        "no_vol_scaler",
        "signal_only",
    }
    assert all(float(row["ci_95_upper"]) < 0.0 for row in ablation_intervals)
    full_interval = next(
        row for row in ablation_intervals if row["variant"] == "full"
    )
    with (PAPER_ROOT / "results" / "bootstrap_sensitivity.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        gross_primary = next(
            row
            for row in csv.DictReader(handle)
            if row["comparison"]
            == "strategy_gross_minus_exposure_matched_equal_weight"
            and row["block_length"] == "21"
        )
    for field in ("annualized_mean", "ci_95_lower", "ci_95_upper"):
        assert full_interval[field] == gross_primary[field]

    with (PAPER_ROOT / "results" / "sharpe_uncertainty.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        sharpe_rows = list(csv.DictReader(handle))
    primary_costed = next(
        row
        for row in sharpe_rows
        if row["series"]
        == "strategy_net_minus_target_exposure_costed_comparator"
        and row["block_length"] == "21"
    )
    assert float(primary_costed["sample_sharpe"]) < 0.0
    assert float(primary_costed["ci_95_upper"]) < 0.0


def test_paper_source_and_reviewed_pdfs_match_release_contract():
    main = (PAPER_ROOT / "main.tex").read_text(encoding="utf-8")
    release_script = (REPO_ROOT / "scripts" / "release_paper_artifacts.sh").read_text(
        encoding="utf-8"
    )

    assert "\\usepackage[dblblindworkshop,final]{neurips_2026}" in main
    assert "\\workshoptitle{TAE (Trust-AI-Eval): Can We Trust AI Evaluation?}" in main
    assert "\\usepackage{orcidlink}" in main
    assert "\\input{results/generated_values}" in main
    assert "\\input{expansion/results/generated_values}" in main
    assert (
        "Auditing Financial ML Backtests with Evaluation Contracts and Executable Checks"
        in main
    )
    assert (
        r"Kevin Lee\,{\hypersetup{pdfborder={0 0 0}}\orcidlink{0009-0004-0388-9260}}"
        in main
    )
    assert "University of California at Los Angeles" in main
    assert "\\url{https://orcid.org/0009-0004-0388-9260}" not in main
    assert "\\input{checklist}" in main
    # Keep the exact input digest in the manifests, outside the paper text.
    assert "\\PaperInputDigest" not in main
    assert "\\And" in main
    assert "{bootstrap_robustness.pdf}" in main
    assert "{contract_effects_return.pdf}" in main
    assert "{baseline_performance.pdf}" in main
    assert (
        "a separate portfolio follows target exposure and pays its own costs"
        in main
    )
    assert "actual_input_digest" in release_script
    assert "expected_input_digest" in release_script
    assert "ls-files --error-unmatch" in release_script
    assert "ls-files --others --exclude-standard" in release_script
    assert release_script.rindex(
        '"${repo_root}/paper/build_manifest.json"'
    ) > release_script.rindex(
        '"${repo_root}/output/pdf/quantcortex_tae_camera_ready.pdf"'
    )
    assert "reviewed_generated_at" in release_script
    assert "reviewed_source_commit" in release_script
    assert "release_source_paths" in release_script
    assert "paper/submission.tex" in release_script
    assert "quantcortex_tae_camera_ready.pdf" in release_script
    assert "scripts/release_paper_artifacts.sh" in release_script
    assert "QUANTCORTEX_GENERATED_AT is required for changed release source" in (
        release_script
    )
    assert '--data-provider "${provider}"' in release_script

    anonymous_source = (PAPER_ROOT / "anonymous.tex").read_text(encoding="ascii")
    assert "\\def\\quantcortexanonymous{1}" in anonymous_source

    submission_source = (PAPER_ROOT / "submission.tex").read_text(encoding="ascii")
    assert "\\def\\quantcortexsubmission{1}" in submission_source
    build_script = (REPO_ROOT / "scripts" / "build_paper.sh").read_text(
        encoding="utf-8"
    )
    assert '"${build_dir}/submission.pdf" "TAE camera-ready" 10' in build_script
    assert "check_review_anonymity" in build_script
    assert "check_camera_ready_identity" in build_script
    assert "contains an exact commit or content fingerprint" in build_script
    assert "quantcortex_tae_camera_ready.pdf" in build_script

    for stem in ("quantcortex_audit_neurips2026", "quantcortex_audit_anonymous"):
        pdf = PAPER_ROOT / f"{stem}.pdf"
        assert pdf.is_file()
        assert pdf.stat().st_size > 100_000
        assert pdf.read_bytes().startswith(b"%PDF-")
        digest, file_name = (PAPER_ROOT / f"{stem}.sha256").read_text(
            encoding="ascii"
        ).split()
        assert file_name == pdf.name
        assert digest == _sha256(pdf)

    pdftotext = shutil.which("pdftotext")
    if pdftotext is not None:
        anonymous_text = subprocess.run(
            [
                pdftotext,
                str(PAPER_ROOT / "quantcortex_audit_anonymous.pdf"),
                "-",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for identifying_text in (
            "Kevin Lee",
            "kevinlee69720",
            "University of California",
            "micro1",
            "github.com/magnaquant",
            "Submitted to",
        ):
            assert identifying_text not in anonymous_text

    build_manifest = json.loads(
        (PAPER_ROOT / "build_manifest.json").read_text(encoding="ascii")
    )
    assert build_manifest["schema_version"] == 2
    assert build_manifest["tectonic_version"] == "0.16.9"
    assert build_manifest["tectonic_bundle"] == {
        "name": "default_bundle_v33.tar",
        "sha256": "6ffe055852f8faf66c0acbe1a7fb27f87b869a90bad1204f3bf4d9683f597c7c",
    }
    assert build_manifest["source_commit"] == json.loads(
        (PAPER_ROOT / "results" / "manifest.json").read_text(encoding="utf-8")
    )["generator"]["git"]["source_commit"]
    assert build_manifest["path_base"] == "repository_root"
    assert build_manifest["pdf"]["sha256"] == _sha256(
        REPO_ROOT / build_manifest["pdf"]["path"]
    )
    assert build_manifest["anonymous_pdf"]["sha256"] == _sha256(
        REPO_ROOT / build_manifest["anonymous_pdf"]["path"]
    )
    assert build_manifest["submission_pdf"]["path"] == (
        "output/pdf/quantcortex_tae_camera_ready.pdf"
    )
    assert build_manifest["submission_pdf"]["tracked"] is False
    assert re.fullmatch(r"[0-9a-f]{64}", build_manifest["submission_pdf"]["sha256"])

    source_manifest = REPO_ROOT / build_manifest["source_manifest"]["path"]
    assert build_manifest["source_manifest"]["entry_path_base"] == "repository_root"
    assert build_manifest["source_manifest"]["sha256"] == _sha256(source_manifest)
    source_entries = {}
    for line in source_manifest.read_text(encoding="ascii").splitlines():
        expected_digest, relative_path = line.split(maxsplit=1)
        source_entries[relative_path] = expected_digest
    assert {
        "paper/main.tex",
        "paper/anonymous.tex",
        "paper/submission.tex",
        "paper/checklist.tex",
        "paper/references.bib",
        "paper/neurips_2026.sty",
        "paper/preregistration.md",
        "paper/results/generated_values.tex",
        "paper/results/manifest.json",
        "paper/expansion/protocol.json",
        "paper/expansion/results/generated_values.tex",
        "paper/expansion/results/manifest.json",
        "paper/figures/accounting_summary.pdf",
        "paper/figures/audit_protocol.pdf",
        "paper/figures/bootstrap_robustness.pdf",
        "paper/figures/engine_comparison.pdf",
        "paper/figures/return_attribution_and_protocol_switches.pdf",
        "paper/figures/sensitivity_and_ablation.pdf",
        "paper/expansion/figures/baseline_performance.pdf",
        "paper/expansion/figures/contract_effects_return.pdf",
        "paper/expansion/figures/contract_effects_sharpe.pdf",
        "paper/expansion/figures/engine_conformance.pdf",
        "paper/expansion/figures/learned_seed_sensitivity.pdf",
        "paper/expansion/figures/yearly_performance.pdf",
        "paper/expansion/figures/seed_contract_effects.pdf",
        "scripts/build_paper.sh",
        "scripts/publish_artifact_bundle.py",
    } == set(source_entries)
    for relative_path, expected_digest in source_entries.items():
        assert _sha256(REPO_ROOT / relative_path) == expected_digest

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "paper/quantcortex_audit_neurips2026.pdf" in readme
    assert "paper/quantcortex_audit_anonymous.pdf" in readme
    assert "paper/figures/return_attribution_and_protocol_switches.png" in readme
    assert "paper/figures/sensitivity_and_ablation.png" in readme
    assert "docs/architecture.md" in readme


def test_locally_built_submission_matches_manifest():
    build_manifest = json.loads(
        (PAPER_ROOT / "build_manifest.json").read_text(encoding="ascii")
    )
    submission = build_manifest.get("submission_pdf")
    if submission is None:
        pytest.skip("tracked release predates the TAE camera-ready build")
    submission_pdf = REPO_ROOT / submission["path"]
    if not submission_pdf.is_file():
        pytest.skip("TAE camera-ready PDF is an ignored build output")
    assert submission_pdf.stat().st_size > 100_000
    assert submission_pdf.read_bytes().startswith(b"%PDF-")
    assert submission["sha256"] == _sha256(submission_pdf)


@pytest.mark.parametrize(
    ("content_pages", "following_page", "within_limit"),
    [
        ("Conclusion", "Acknowledgments and Disclosure of Funding\nDisclosure", True),
        ("Conclusion", "\f  120 Acknowledgments and Disclosure of Funding\n11", True),
        ("Conclusion\nAcknowledgments and Disclosure of Funding", "Disclosure", True),
        ("Conclusion\nReferences", "More references", True),
        ("Conclusion", "References\nFirst citation", True),
        (
            "Conclusion",
            "Remaining conclusion\nAcknowledgments and Disclosure of Funding\nReferences",
            False,
        ),
        ("Conclusion", "Remaining conclusion\nReferences", False),
        ("Conclusion", "We discuss Acknowledgments and Disclosure of Funding.", False),
        ("Conclusion", "Remaining conclusion", False),
    ],
)
def test_build_body_limit_accounts_for_acknowledgments(
    monkeypatch, content_pages, following_page, within_limit
):
    script = (REPO_ROOT / "scripts/build_paper.sh").read_text()
    body_check = "check_body_limit() {" + script.split(
        "check_body_limit() {", maxsplit=1
    )[1].split("\nfile_matches() {", maxsplit=1)[0]
    monkeypatch.setenv("QUANTCORTEX_TEST_BODY_PAGES", content_pages)
    monkeypatch.setenv("QUANTCORTEX_TEST_FOLLOWING_PAGE", following_page)
    result = subprocess.run(
        [
            "bash",
            "-c",
            """set -euo pipefail
pdftotext() {
  if [[ "$2" == "1" ]]; then
    printf '%s\\n' "$QUANTCORTEX_TEST_BODY_PAGES"
  else
    printf '%s\\n' "$QUANTCORTEX_TEST_FOLLOWING_PAGE"
  fi
}
"""
            + body_check
            + '\ncheck_body_limit unused.pdf "fixture paper" 10\n',
        ],
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == within_limit, result.stderr
    if not within_limit:
        assert "exceeds its 10-page content limit" in result.stderr


def test_reviewed_paper_body_respects_build_page_limits():
    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        import pytest

        pytest.skip("pdftotext is required to inspect PDF page boundaries")

    def page_text(pdf: Path, first: int, last: int) -> str:
        return subprocess.run(
            [
                pdftotext,
                "-f",
                str(first),
                "-l",
                str(last),
                "-layout",
                pdf,
                "-",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    for pdf_name, page_limit in (
        ("quantcortex_audit_neurips2026.pdf", 10),
        ("quantcortex_audit_anonymous.pdf", 10),
    ):
        pdf = PAPER_ROOT / pdf_name
        content_pages = page_text(pdf, 1, page_limit)
        following_page = page_text(pdf, page_limit + 1, page_limit + 1)
        boundary_pattern = re.compile(
            r"^\s*(?:\d+\s+)?(?:Acknowledgments and Disclosure of Funding|References)\s*$",
            re.MULTILINE,
        )
        if boundary_pattern.search(content_pages):
            continue

        substantive_lines = [
            re.sub(r"^\d+\s+", "", line.strip())
            for line in following_page.replace("\f", "").splitlines()
            if line.strip() and not line.strip().isdigit()
        ]
        assert substantive_lines, pdf_name
        assert substantive_lines[0] in {
            "Acknowledgments and Disclosure of Funding", "References"
        }, pdf_name


def test_submission_source_requires_tae_camera_ready_limit_and_float_barrier():
    main = (PAPER_ROOT / "main.tex").read_text(encoding="utf-8")
    build_script = (REPO_ROOT / "scripts" / "build_paper.sh").read_text(
        encoding="utf-8"
    )

    abstract = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main, re.DOTALL
    )
    assert abstract is not None
    assert re.search(r"\n\s*\n", abstract.group(1).strip()) is None
    main_body = main.split("\\bibliographystyle{plainnat}", maxsplit=1)[0]
    for table in re.findall(
        r"\\begin\{table\}.*?\\end\{table\}", main_body, re.DOTALL
    ):
        assert "\\small" not in table

    assert "\\usepackage{placeins}" in main
    bibliography = main.index("\\bibliographystyle{plainnat}")
    assert "\\FloatBarrier" in main[:bibliography][-100:]
    assert (
        'check_body_limit "${build_dir}/submission.pdf" "TAE camera-ready" 10'
        in build_script
    )
    assert 'check_body_limit "${build_dir}/main.pdf" "public paper" 10' in build_script
    assert "pdftotext is required for paper validation" in build_script
    assert "pdfinfo is required for paper validation" in build_script
    assert "pdfdetach is required for paper validation" in build_script
    assert "pdffonts is required for paper validation" in build_script
    assert "strings is required for paper validation" in build_script
    assert "contains embedded files" in build_script
    assert "has nonempty {field.lower()} metadata" in build_script
    assert "Type 3 or unembedded font" in build_script
    assert 'pdftotext -layout "${pdf_path}" - &&' in build_script
    assert 'source_hash="$(sha256_file' in build_script
    assert 'source_manifest_hash="$(sha256_file' in build_script
    assert "file_matches" in build_script
    assert "has content outside the text rectangle" in build_script
    assert '"path_base": "repository_root"' in build_script
    publish_start = build_script.index(
        "# Install the complete validated bundle or restore every previous destination."
    )
    assert 'check_review_anonymity "${build_dir}/submission.pdf"' not in build_script
    assert build_script.index(
        'check_review_anonymity "${build_dir}/anonymous.pdf"'
    ) < publish_start
    assert build_script.index(
        'check_camera_ready_identity "${build_dir}/submission.pdf"'
    ) < publish_start
    assert build_script.rindex('"${paper_dir}/build_manifest.json"') > publish_start
    assert 'source_manifest="${build_dir}/' in build_script
    assert 'prebuild_source_manifest="${build_dir}/' in build_script
    assert 'source_snapshot="${build_dir}/source"' in build_script
    assert 'build_dir="$(mktemp -d "${build_root}/build.XXXXXX")"' in build_script
    assert 'cd "${source_snapshot}/paper"' in build_script
    assert 'publisher_path="${source_snapshot}/scripts/publish_artifact_bundle.py"' in (
        build_script
    )
    assert 'flock -x "${publication_lock_fd}"' in build_script
    assert "paper sources changed during the build" in build_script
    assert "paper sources changed while the build snapshot was created" in build_script
    assert '"entry_path_base": "repository_root"' in build_script
    assert '"tracked": False' in build_script
    assert 'build_manifest="${build_dir}/build_manifest.json"' in build_script


@pytest.mark.parametrize(
    ("review_text", "scanner_error_pattern", "expected_error"),
    [
        ("Jimmy Chan\nKevin Lee\nTAE (Trust-AI-Eval)", "", ""),
        (
            "Kevin Lee\nTAE (Trust-AI-Eval)",
            "",
            "is missing the identified author Jimmy Chan",
        ),
        (
            "Jimmy Chan\nTAE (Trust-AI-Eval)",
            "",
            "is missing the identified author Kevin Lee",
        ),
        ("Jimmy Chan\nKevin Lee", "", "is missing the TAE workshop title"),
        (
            "Jimmy Chan\nKevin Lee\nTAE (Trust-AI-Eval)\nFoundation Models for Temporal Systems",
            "",
            "still names the FMTS workshop",
        ),
        (
            "Jimmy Chan\nKevin Lee\nTAE (Trust-AI-Eval)\nAnonymous author",
            "",
            "is still anonymous",
        ),
        *[
            (
                "Jimmy Chan\nKevin Lee\nTAE (Trust-AI-Eval)",
                pattern,
                "validation scanner failed with status 2",
            )
            for pattern in (
                "Jimmy Chan",
                "Kevin Lee",
                "TAE (Trust-AI-Eval)",
                "Foundation Models for Temporal Systems",
                "Anonymous author",
            )
        ],
    ],
)
def test_camera_ready_identity_shell_checks(
    tmp_path, review_text, scanner_error_pattern, expected_error
):
    build_script = (REPO_ROOT / "scripts" / "build_paper.sh").read_text(
        encoding="utf-8"
    )
    functions = []
    for name in ("file_matches", "check_camera_ready_identity"):
        match = re.search(
            rf"^{name}\(\) \{{\n.*?^\}}", build_script, re.MULTILINE | re.DOTALL
        )
        assert match is not None, name
        functions.append(match.group())

    # Run only the real validation functions, never the build entry point.
    # Scanner injection and PDF text substitution are local to this child shell.
    shell = r'''
set -euo pipefail
review_text="$1"
scanner_error_pattern="$2"
pdftotext() { printf '%s\n' "${review_text}"; }
grep() {
  local arg
  for arg in "$@"; do
    if [[ -n "${scanner_error_pattern}" && "${arg}" == "${scanner_error_pattern}" ]]; then
      return 2
    fi
  done
  command grep "$@"
}
'''
    result = subprocess.run(
        [
            "bash",
            "-c",
            shell + "\n".join(functions)
            + '\ncheck_camera_ready_identity "submission.pdf" "TAE camera-ready"\n',
            "identity-test",
            review_text,
            scanner_error_pattern,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == (1 if expected_error else 0), result.stderr
    assert result.stderr == (
        f"TAE camera-ready {expected_error}\n" if expected_error else ""
    )
    assert result.stdout == ""
    assert (tmp_path / "submission.camera.txt").read_text(encoding="utf-8") == (
        review_text + "\n"
    )


@pytest.mark.parametrize(
    "identifier",
    [
        "Jimmy Chan",
        "Pepperdine University",
        "jimmybojim.chan@alumnimail.pepperdine.edu",
        "0009-0002-4670-2828",
        "Kevin Lee",
        "kevinlee69720@g.ucla.edu",
        "0009-0004-0388-9260",
        "",
    ],
)
def test_review_anonymity_shell_checks_both_authors(tmp_path, identifier):
    build_script = (REPO_ROOT / "scripts" / "build_paper.sh").read_text(
        encoding="utf-8"
    )
    functions = []
    for name in ("file_matches", "check_review_anonymity"):
        match = re.search(
            rf"^{name}\(\) \{{\n.*?^\}}", build_script, re.MULTILINE | re.DOTALL
        )
        assert match is not None, name
        functions.append(match.group())

    # Exercise the real guard with controlled PDF inspection output.
    shell = r'''
set -euo pipefail
review_text="$1"
python_bin="$2"
pdfdetach() { printf '0 embedded files\n'; }
pdfinfo() { printf 'Pages: 1\n'; }
pdftotext() { printf '%s\n' "${review_text}"; }
strings() { :; }
'''
    result = subprocess.run(
        [
            "bash",
            "-c",
            shell + "\n".join(functions)
            + '\ncheck_review_anonymity "anonymous.pdf" "anonymous paper"\n',
            "anonymity-test",
            identifier,
            sys.executable,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == (1 if identifier else 0), result.stderr
    assert result.stderr == (
        "anonymous paper contains an author or repository identifier\n"
        if identifier else ""
    )
    assert result.stdout == ""


def test_paper_citations_are_defined_used_and_unique():
    main = (PAPER_ROOT / "main.tex").read_text(encoding="utf-8")
    bibliography = (PAPER_ROOT / "references.bib").read_text(encoding="utf-8")

    cited_keys = {
        key.strip()
        for group in re.findall(r"\\cite\w*\{([^}]*)\}", main)
        for key in group.split(",")
    }
    bibliography_keys = set(re.findall(r"@\w+\{([^,]+),", bibliography))

    assert cited_keys == bibliography_keys
    assert "{L{\\'o}pez de Prado}, Marcos" in bibliography
    assert "{d'Alch{\\'e}-Buc}, Florence" in bibliography
    assert "The Statistics of {Sharpe} Ratios" in bibliography
    assert "The Deflated {Sharpe} Ratio" in bibliography

    dois = [
        value.lower()
        for value in re.findall(r"doi\s*=\s*\{([^}]+)\}", bibliography)
    ]
    eprints = re.findall(r"eprint\s*=\s*\{([^}]+)\}", bibliography)
    assert len(dois) == len(set(dois))
    assert len(eprints) == len(set(eprints))

    for eprint in eprints:
        assert f"https://arxiv.org/abs/{eprint}" in bibliography


def _build_expansion_fixture(root: Path) -> Path:
    """Create synthetic files with matching digests for artifact-verifier tests."""
    expansion = root / "expansion"
    (expansion / "results").mkdir(parents=True)
    (expansion / "figures").mkdir()
    (expansion / "results" / "summary.csv").write_text("a,b\n1,2\n", encoding="ascii")
    (expansion / "figures" / "plot.pdf").write_bytes(b"%PDF-1.4 fixture")
    artifacts = {
        "results/summary.csv": _sha256(expansion / "results" / "summary.csv"),
        "figures/plot.pdf": _sha256(expansion / "figures" / "plot.pdf"),
    }
    manifest = {"git": {"source_commit": "0" * 40}, "artifacts": artifacts}
    (expansion / "results" / "manifest.json").write_text(
        json.dumps(manifest), encoding="ascii"
    )
    return expansion


def test_verify_expansion_artifacts_accepts_valid_tree(tmp_path):
    expansion = _build_expansion_fixture(tmp_path)
    verify_expansion_artifacts(expansion)


def test_verify_expansion_artifacts_rejects_tampered_artifact(tmp_path):
    # Checking only the manifest's source commit would let a tampered aggregate
    # reach the paper build without an artifact-digest failure.
    expansion = _build_expansion_fixture(tmp_path)
    (expansion / "results" / "summary.csv").write_text("a,b\n9,9\n", encoding="ascii")
    with pytest.raises(SystemExit, match="digest mismatch"):
        verify_expansion_artifacts(expansion)


def test_verify_expansion_artifacts_rejects_missing_artifact(tmp_path):
    expansion = _build_expansion_fixture(tmp_path)
    (expansion / "figures" / "plot.pdf").unlink()
    with pytest.raises(SystemExit, match="missing"):
        verify_expansion_artifacts(expansion)


def test_verify_expansion_artifacts_rejects_unexpected_file(tmp_path):
    expansion = _build_expansion_fixture(tmp_path)
    (expansion / "results" / "extra.csv").write_text("x\n", encoding="ascii")
    with pytest.raises(SystemExit, match="unexpected expansion artifact"):
        verify_expansion_artifacts(expansion)


def test_verify_expansion_artifacts_rejects_symlink(tmp_path):
    expansion = _build_expansion_fixture(tmp_path)
    target = expansion / "results" / "summary.csv"
    link = expansion / "figures" / "plot.pdf"
    link.unlink()
    link.symlink_to(target)
    with pytest.raises(SystemExit, match="symlink"):
        verify_expansion_artifacts(expansion)


def test_verify_expansion_artifacts_rejects_unsafe_manifest_paths(tmp_path):
    expansion = _build_expansion_fixture(tmp_path)
    manifest_path = expansion / "results" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))

    outside = tmp_path / "outside.txt"
    outside.write_text("leak\n", encoding="ascii")
    for unsafe in ("../outside.txt", str(outside)):
        tampered = dict(manifest)
        tampered["artifacts"] = dict(manifest["artifacts"])
        tampered["artifacts"][unsafe] = _sha256(outside)
        manifest_path.write_text(json.dumps(tampered), encoding="ascii")
        with pytest.raises(SystemExit, match="unsafe"):
            verify_expansion_artifacts(expansion)


def test_verify_expansion_artifacts_rejects_empty_manifest(tmp_path):
    expansion = _build_expansion_fixture(tmp_path)
    manifest_path = expansion / "results" / "manifest.json"
    manifest_path.write_text(json.dumps({"artifacts": {}}), encoding="ascii")
    with pytest.raises(SystemExit, match="lists no artifacts"):
        verify_expansion_artifacts(expansion)


@pytest.mark.parametrize("alias_kind", ["parent_component", "symlink"])
def test_artifact_bundle_rejects_destination_aliases(tmp_path, alias_kind):
    source = tmp_path / "source.txt"
    source.write_text("new\n", encoding="ascii")
    directory = tmp_path / "published"
    directory.mkdir()
    destination = directory / "value.txt"
    destination.write_text("old\n", encoding="ascii")
    if alias_kind == "parent_component":
        child = directory / "child"
        child.mkdir()
        alias = child / ".." / "value.txt"
    else:
        alias_directory = tmp_path / "alias"
        alias_directory.symlink_to(directory, target_is_directory=True)
        alias = alias_directory / "value.txt"

    with pytest.raises(ValueError, match="unique"):
        publish_artifact_bundle.publish_bundle([(source, destination), (source, alias)])

    assert destination.read_text(encoding="ascii") == "old\n"
    assert not list(tmp_path.rglob(".*.quantcortex-*"))


@pytest.mark.parametrize("reverse", [False, True])
def test_artifact_bundle_rejects_nested_destinations(tmp_path, reverse):
    source = tmp_path / "source"
    source.mkdir()
    (source / "value.txt").write_text("new\n", encoding="ascii")
    destination = tmp_path / "published"
    destination.mkdir()
    child = destination / "value.txt"
    child.write_text("old\n", encoding="ascii")
    entries = [(source, destination), (source / "value.txt", child)]
    if reverse:
        entries.reverse()

    with pytest.raises(ValueError, match="contain one another"):
        publish_artifact_bundle.publish_bundle(entries)

    assert child.read_text(encoding="ascii") == "old\n"
    assert list(destination.iterdir()) == [child]
    assert not list(tmp_path.rglob(".*.quantcortex-*"))


def test_artifact_bundle_replaces_final_symlink_without_changing_target(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("new\n", encoding="ascii")
    target = tmp_path / "target.txt"
    target.write_text("original\n", encoding="ascii")
    destination = tmp_path / "published.txt"
    destination.symlink_to(target)

    publish_artifact_bundle.publish_bundle([(source, destination)])

    assert not destination.is_symlink()
    assert destination.read_text(encoding="ascii") == "new\n"
    assert target.read_text(encoding="ascii") == "original\n"


def test_artifact_bundle_publishes_files_and_directories(tmp_path):
    source_file = tmp_path / "new.txt"
    source_file.write_text("new file\n", encoding="ascii")
    source_directory = tmp_path / "new-directory"
    source_directory.mkdir()
    (source_directory / "value.txt").write_text("new directory\n", encoding="ascii")
    destination_file = tmp_path / "published.txt"
    destination_file.write_text("old file\n", encoding="ascii")
    destination_directory = tmp_path / "published-directory"
    destination_directory.mkdir()
    (destination_directory / "value.txt").write_text(
        "old directory\n", encoding="ascii"
    )

    publish_artifact_bundle.publish_bundle(
        [
            (source_file, destination_file),
            (source_directory, destination_directory),
        ],
        lock_path=tmp_path / "publication.lock",
    )

    assert destination_file.read_text(encoding="ascii") == "new file\n"
    assert (destination_directory / "value.txt").read_text(
        encoding="ascii"
    ) == "new directory\n"
    assert not list(tmp_path.glob(".*.quantcortex-*.tmp"))
    assert not list(tmp_path.glob(".*.quantcortex-*.backup"))


def test_artifact_bundle_rolls_back_partial_install(tmp_path, monkeypatch):
    source_file = tmp_path / "new.txt"
    source_file.write_text("new file\n", encoding="ascii")
    source_directory = tmp_path / "new-directory"
    source_directory.mkdir()
    (source_directory / "value.txt").write_text("new directory\n", encoding="ascii")
    destination_file = tmp_path / "published.txt"
    destination_file.write_text("old file\n", encoding="ascii")
    destination_directory = tmp_path / "published-directory"
    destination_directory.mkdir()
    (destination_directory / "value.txt").write_text(
        "old directory\n", encoding="ascii"
    )
    real_replace = publish_artifact_bundle.os.replace
    replace_count = 0

    def fail_second_install(source, destination):
        nonlocal replace_count
        replace_count += 1
        if replace_count == 4:
            raise OSError("injected second-install failure")
        return real_replace(source, destination)

    monkeypatch.setattr(publish_artifact_bundle.os, "replace", fail_second_install)

    with pytest.raises(OSError, match="injected second-install failure"):
        publish_artifact_bundle.publish_bundle(
            [
                (source_file, destination_file),
                (source_directory, destination_directory),
            ]
        )

    assert destination_file.read_text(encoding="ascii") == "old file\n"
    assert (destination_directory / "value.txt").read_text(
        encoding="ascii"
    ) == "old directory\n"
    assert not list(tmp_path.glob(".*.quantcortex-*.tmp"))
    assert not list(tmp_path.glob(".*.quantcortex-*.backup"))


def test_artifact_bundle_rolls_back_interrupt_after_rename(tmp_path, monkeypatch):
    first_source = tmp_path / "first-new.txt"
    first_source.write_text("first new\n", encoding="ascii")
    second_source = tmp_path / "second-new.txt"
    second_source.write_text("second new\n", encoding="ascii")
    first_destination = tmp_path / "first-published.txt"
    first_destination.write_text("first old\n", encoding="ascii")
    second_destination = tmp_path / "second-published.txt"
    real_replace = publish_artifact_bundle.os.replace
    replace_count = 0

    def interrupt_after_second_install(source, destination):
        nonlocal replace_count
        replace_count += 1
        real_replace(source, destination)
        if replace_count == 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(
        publish_artifact_bundle.os,
        "replace",
        interrupt_after_second_install,
    )

    with pytest.raises(KeyboardInterrupt):
        publish_artifact_bundle.publish_bundle(
            [
                (first_source, first_destination),
                (second_source, second_destination),
            ]
        )

    assert first_destination.read_text(encoding="ascii") == "first old\n"
    assert not second_destination.exists()
    assert not list(tmp_path.glob(".*.quantcortex-*.tmp"))
    assert not list(tmp_path.glob(".*.quantcortex-*.backup"))


def test_artifact_bundle_cli_publishes_under_lock(tmp_path):
    source = tmp_path / "new.txt"
    source.write_text("new\n", encoding="ascii")
    destination = tmp_path / "published.txt"
    lock_path = tmp_path / "publication.lock"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "publish_artifact_bundle.py"),
        "--lock",
        str(lock_path),
        str(source),
        str(destination),
    ]

    with lock_path.open("w", encoding="ascii") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        process = subprocess.Popen(command)
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=0.5)
        assert not destination.exists()
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        assert process.wait(timeout=10) == 0

    assert destination.read_text(encoding="ascii") == "new\n"


def test_expansion_release_rejects_tracked_raw_panels(tmp_path):
    repository = tmp_path / "repository"
    source_files = [
        "quantcortex/__init__.py",
        "schemas/canonical_target_tape.schema.json",
        "pyproject.toml",
        "poetry.lock",
        "paper/preregistration.md",
        "paper/expansion/protocol.json",
        "scripts/fetch_expansion_data.py",
        "scripts/run_expansion_experiments.py",
    ]
    for relative_path in source_files:
        path = repository / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="ascii")
    scripts = repository / "scripts"
    shutil.copy2(
        REPO_ROOT / "scripts" / "release_expansion_artifacts.sh",
        scripts / "release_expansion_artifacts.sh",
    )
    shutil.copy2(
        REPO_ROOT / "scripts" / "publish_artifact_bundle.py",
        scripts / "publish_artifact_bundle.py",
    )
    python_path = repository / ".venv" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.symlink_to(sys.executable)
    panel_dir = repository / "local_data" / "expansion"
    panel_dir.mkdir(parents=True)
    for panel in ("us_sector_etfs", "country_equity_etfs"):
        (panel_dir / f"{panel}.csv").write_text("Date,SHV\n", encoding="ascii")
        (panel_dir / f"{panel}.metadata.json").write_text("{}\n", encoding="ascii")

    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )

    result = subprocess.run(
        [
            "bash",
            str(scripts / "release_expansion_artifacts.sh"),
            str(panel_dir),
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert (
        "reviewed raw expansion input must not be tracked by Git: "
        "local_data/expansion/us_sector_etfs.csv"
    ) in result.stderr


def test_release_wrapper_verifies_copied_expansion_artifacts_before_building():
    release_script = (REPO_ROOT / "scripts" / "release_paper_artifacts.sh").read_text(
        encoding="utf-8"
    )
    expansion_release_script = (
        REPO_ROOT / "scripts" / "release_expansion_artifacts.sh"
    ).read_text(encoding="utf-8")
    assert "ls-files --others --exclude-standard" in release_script
    assert "ls-files --others --exclude-standard" in expansion_release_script
    assert "ls-files --error-unmatch" in release_script
    assert "ls-files --error-unmatch" in expansion_release_script
    assert "publish_bundle()" in release_script
    assert "publish_bundle()" in expansion_release_script
    assert "publish_artifact_bundle.py" in release_script
    assert "publish_artifact_bundle.py" in expansion_release_script
    assert 'flock -x "${publication_lock_fd}"' in release_script
    assert 'flock -x "${publication_lock_fd}"' in expansion_release_script
    assert 'publisher_path="${source_worktree}/scripts/publish_artifact_bundle.py"' in (
        release_script
    )
    assert 'publisher_path="${source_worktree}/scripts/publish_artifact_bundle.py"' in (
        expansion_release_script
    )
    assert "verify_expansion_artifacts" in release_script
    # Verify the detached-worktree copy, not an earlier view of mutable source.
    assert release_script.index(
        'cp -R "${repo_root}/paper/expansion/results"'
    ) < release_script.index("verify_expansion_artifacts")
    assert release_script.index("verify_expansion_artifacts") < release_script.rindex(
        "scripts/build_paper.sh"
    )
    assert '"${input_snapshot}"' in release_script
    assert "emitted_input_digest" in release_script
    assert 'input_snapshot="${temporary_root}/inputs"' in expansion_release_script
    assert "expansion input matrices do not match the reviewed release" in (
        expansion_release_script
    )
    assert "expansion data provenance changed from the reviewed release" in (
        expansion_release_script
    )
    assert "QUANTCORTEX_EXPECTED_INPUT_SHA256" in release_script
    for variable in (
        "QUANTCORTEX_DATA_PROVIDER",
        "QUANTCORTEX_PERMISSION_BASIS",
        "QUANTCORTEX_RETRIEVED_AT",
        "QUANTCORTEX_ADJUSTMENT_METHOD",
    ):
        assert variable in release_script
    assert '"${variable} is required for an input refresh"' in release_script
    assert "QUANTCORTEX_EXPANSION_US_SECTOR_INPUT_SHA256" in (
        expansion_release_script
    )
    assert "QUANTCORTEX_EXPANSION_COUNTRY_INPUT_SHA256" in (
        expansion_release_script
    )
    assert "expansion input matrices do not match explicit refresh digests" in (
        expansion_release_script
    )


def test_committed_expansion_artifacts_validate():
    verify_expansion_artifacts(PAPER_ROOT / "expansion")
