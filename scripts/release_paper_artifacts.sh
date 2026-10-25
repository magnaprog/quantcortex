#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"
prices_argument="${1:-${repo_root}/local_data/published_rotation_prices.csv}"
publisher_path="${repo_root}/scripts/publish_artifact_bundle.py"

publish_bundle() {
  "${python_bin}" "${publisher_path}" "$@"
}

if [[ ! -x "${python_bin}" ]]; then
  printf '%s\n' "Python environment not found: ${python_bin}" >&2
  exit 1
fi
command -v flock >/dev/null 2>&1 || {
  printf '%s\n' "flock is required for artifact publication" >&2
  exit 1
}
mkdir -p "${repo_root}/tmp"
exec {publication_lock_fd}>"${repo_root}/tmp/artifact-publication.lock"
flock -x "${publication_lock_fd}"
export QUANTCORTEX_PUBLICATION_LOCK_HELD=1

# Generation runs from a detached worktree at the source commit, so
# uncommitted changes outside release-critical source cannot alter what is
# generated. Scoping the cleanliness check to those paths lets the paper
# release run after the expansion wrapper has deposited regenerated artifacts
# in the working tree, which the documented dual-release flow requires. The
# expansion artifacts that this release republishes are separately verified
# against the expansion manifest below before they are copied.
release_source_paths=(
  quantcortex
  schemas
  pyproject.toml
  poetry.lock
  scripts/build_paper.sh
  scripts/generate_report.py
  scripts/fetch_expansion_data.py
  scripts/publish_artifact_bundle.py
  scripts/release_expansion_artifacts.sh
  scripts/release_paper_artifacts.sh
  scripts/run_expansion_experiments.py
  scripts/run_paper_experiments.py
  paper/main.tex
  paper/audit_protocol.drawio
  paper/audit_protocol.drawio.pdf
  paper/audit_protocol.drawio.png
  paper/anonymous.tex
  paper/submission.tex
  paper/checklist.tex
  paper/references.bib
  paper/neurips_2026.sty
  paper/preregistration.md
  paper/expansion/protocol.json
)
for release_source_path in "${release_source_paths[@]}"; do
  git -C "${repo_root}" ls-files --error-unmatch "${release_source_path}" \
    >/dev/null 2>&1 || {
      printf '%s\n' \
        "release-critical source is not tracked: ${release_source_path}" >&2
      exit 1
    }
done
untracked_release_source="$(
  git -C "${repo_root}" ls-files --others --exclude-standard -- \
    "${release_source_paths[@]}"
)"
if [[ -n "${untracked_release_source}" ]]; then
  printf '%s\n' \
    "release-critical source is untracked:" \
    "${untracked_release_source}" >&2
  exit 1
fi
if ! git -C "${repo_root}" diff --quiet -- "${release_source_paths[@]}" || \
   ! git -C "${repo_root}" diff --cached --quiet -- "${release_source_paths[@]}"; then
  printf '%s\n' \
    "commit release-critical source changes before releasing paper artifacts" >&2
  exit 1
fi

prices_csv="$(
  "${python_bin}" -c \
    'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' \
    "${prices_argument}"
)"
if [[ ! -f "${prices_csv}" ]]; then
  printf '%s\n' "price matrix not found: ${prices_csv}" >&2
  exit 1
fi

reviewed_manifest="${repo_root}/paper/results/manifest.json"
if [[ ! -f "${reviewed_manifest}" ]]; then
  printf '%s\n' \
    "reviewed paper manifest not found: ${reviewed_manifest}" >&2
  exit 1
fi
refresh_input_digest="${QUANTCORTEX_EXPECTED_INPUT_SHA256:-}"
if [[ -n "${refresh_input_digest}" ]]; then
  if [[ ! "${refresh_input_digest}" =~ ^[0-9a-f]{64}$ ]]; then
    printf '%s\n' \
      "QUANTCORTEX_EXPECTED_INPUT_SHA256 must be a lowercase SHA-256 value" >&2
    exit 1
  fi
  if [[ ! "${QUANTCORTEX_GENERATED_AT:-}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]]; then
    printf '%s\n' \
      "QUANTCORTEX_GENERATED_AT must be a fixed UTC timestamp for an input refresh" >&2
    exit 1
  fi
  refresh_provenance_variables=(
    QUANTCORTEX_DATA_PROVIDER
    QUANTCORTEX_PERMISSION_BASIS
    QUANTCORTEX_RETRIEVED_AT
    QUANTCORTEX_ADJUSTMENT_METHOD
  )
  for variable in "${refresh_provenance_variables[@]}"; do
    if [[ -z "${!variable:-}" ]]; then
      printf '%s\n' "${variable} is required for an input refresh" >&2
      exit 1
    fi
  done
  expected_input_digest="${refresh_input_digest}"
else
  expected_input_digest="$(
    "${python_bin}" -c \
      'import json, sys; print(json.load(open(sys.argv[1]))["source"]["input_sha256"])' \
      "${reviewed_manifest}"
  )"
fi
actual_input_digest="$(
  "${python_bin}" -c \
    'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
    "${prices_csv}"
)"
if [[ "${actual_input_digest}" != "${expected_input_digest}" ]]; then
  printf '%s\n' \
    "price matrix digest does not match the expected release input" \
    "expected: ${expected_input_digest}" \
    "actual:   ${actual_input_digest}" >&2
  exit 1
fi
case "${prices_csv}" in
  "${repo_root}"/*)
    relative_prices="${prices_csv#${repo_root}/}"
    if git -C "${repo_root}" ls-files --error-unmatch -- "${relative_prices}" \
      >/dev/null 2>&1; then
      printf '%s\n' \
        "reviewed raw price input must not be tracked by Git: ${relative_prices}" >&2
      exit 1
    else
      tracking_status=$?
      if [[ "${tracking_status}" -ne 1 ]]; then
        printf '%s\n' \
          "could not determine whether the raw price input is tracked" >&2
        exit 1
      fi
    fi
    ;;
esac

current_commit="$(git -C "${repo_root}" rev-parse HEAD)"
reviewed_source_commit="$(
  "${python_bin}" -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["generator"]["git"]["source_commit"])' \
    "${reviewed_manifest}"
)"
reviewed_generated_at="$(
  "${python_bin}" -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["generated_at"])' \
    "${reviewed_manifest}"
)"
if [[ -n "${QUANTCORTEX_GENERATED_AT:-}" ]]; then
  source_commit="${current_commit}"
  generated_at="${QUANTCORTEX_GENERATED_AT}"
else
  if ! git -C "${repo_root}" cat-file -e "${reviewed_source_commit}^{commit}" \
    >/dev/null 2>&1; then
    printf '%s\n' "reviewed source commit is unavailable" >&2
    exit 1
  fi
  if ! git -C "${repo_root}" diff --quiet \
    "${reviewed_source_commit}" "${current_commit}" -- \
    "${release_source_paths[@]}"; then
    printf '%s\n' \
      "QUANTCORTEX_GENERATED_AT is required for changed release source" >&2
    exit 1
  fi
  source_commit="${reviewed_source_commit}"
  generated_at="${reviewed_generated_at}"
fi
source_date_epoch="${SOURCE_DATE_EPOCH:-$(git -C "${repo_root}" show -s --format=%ct "${source_commit}")}"
temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/quantcortex-paper-release.XXXXXX")"
source_worktree="${temporary_root}/source"
generated_output="${temporary_root}/generated"
performance_output="${temporary_root}/performance"
input_snapshot="${temporary_root}/published_rotation_prices.csv"

cleanup() {
  git -C "${repo_root}" worktree remove --force "${source_worktree}" \
    >/dev/null 2>&1 || true
  rm -rf "${temporary_root}"
}
trap cleanup EXIT

git -C "${repo_root}" worktree add --detach "${source_worktree}" "${source_commit}" \
  >/dev/null
publisher_path="${source_worktree}/scripts/publish_artifact_bundle.py"
cp "${prices_csv}" "${input_snapshot}"
snapshot_input_digest="$(
  "${python_bin}" -c \
    'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
    "${input_snapshot}"
)"
if [[ "${snapshot_input_digest}" != "${expected_input_digest}" ]]; then
  printf '%s\n' "price matrix changed while the release snapshot was created" >&2
  exit 1
fi

if [[ -n "${refresh_input_digest}" ]]; then
  provider="${QUANTCORTEX_DATA_PROVIDER}"
  permission="${QUANTCORTEX_PERMISSION_BASIS}"
  retrieved_at="${QUANTCORTEX_RETRIEVED_AT}"
  adjustment="${QUANTCORTEX_ADJUSTMENT_METHOD}"
else
  provider="$(
    "${python_bin}" -c \
      'import json, sys; print(json.load(open(sys.argv[1]))["source"]["provider"])' \
      "${reviewed_manifest}"
  )"
  permission="$(
    "${python_bin}" -c \
      'import json, sys; print(json.load(open(sys.argv[1]))["source"]["permission_basis"])' \
      "${reviewed_manifest}"
  )"
  retrieved_at="$(
    "${python_bin}" -c \
      'import json, sys; print(json.load(open(sys.argv[1]))["source"]["retrieved_at"])' \
      "${reviewed_manifest}"
  )"
  adjustment="$(
    "${python_bin}" -c \
      'import json, sys; print(json.load(open(sys.argv[1]))["source"]["adjustment_method"])' \
      "${reviewed_manifest}"
  )"
fi

(
  cd "${source_worktree}"
  MPLCONFIGDIR="${temporary_root}/matplotlib" \
  PYTHONPATH="${source_worktree}" \
  "${python_bin}" scripts/run_paper_experiments.py \
    --prices-csv "${input_snapshot}" \
    --cash-proxy-symbol SHV \
    --output-dir "${generated_output}" \
    --bootstrap-replications 5000 \
    --data-provider "${provider}" \
    --permission-basis "${permission}" \
    --retrieved-at "${retrieved_at}" \
    --adjustment-method "${adjustment}" \
    --generated-at "${generated_at}" \
    --require-clean-source
)

(
  cd "${source_worktree}"
  MPLCONFIGDIR="${temporary_root}/matplotlib-report" \
  PYTHONPATH="${source_worktree}" \
  "${python_bin}" scripts/generate_report.py \
    --prices-csv "${input_snapshot}" \
    --cash-proxy-symbol SHV \
    --imgdir "${performance_output}/img" \
    --report-out "${performance_output}/report.md" \
    --manifest-out "${performance_output}/img/performance_manifest.json" \
    --data-provider "${provider}" \
    --permission-basis "${permission}" \
    --retrieved-at "${retrieved_at}" \
    --adjustment-method "${adjustment}" \
    --generated-at "${generated_at}" \
    --require-clean-source \
    >/dev/null
)

"${python_bin}" - \
  "${generated_output}/results/manifest.json" \
  "${source_commit}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_commit = sys.argv[2]
git = manifest["generator"]["git"]
if git["source_commit"] != expected_commit:
    raise SystemExit("manifest source commit does not match the release commit")
if git["worktree_clean_at_start"] is not True:
    raise SystemExit("manifest did not record a clean source worktree")
PY

emitted_input_digest="$(
  "${python_bin}" -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["source"]["input_sha256"])' \
    "${generated_output}/results/manifest.json"
)"
if [[ "${emitted_input_digest}" != "${expected_input_digest}" ]]; then
  printf '%s\n' "generated manifest records an unexpected price matrix" >&2
  exit 1
fi

"${python_bin}" - \
  "${performance_output}/img/performance_manifest.json" \
  "${source_commit}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_commit = sys.argv[2]
git = manifest["generator"]["git"]
if git["source_commit"] != expected_commit:
    raise SystemExit("performance manifest source commit does not match")
if git["worktree_clean_at_start"] is not True:
    raise SystemExit("performance manifest did not record a clean source worktree")
PY

rm -rf "${source_worktree}/paper/results" "${source_worktree}/paper/figures"
cp -R "${generated_output}/results" "${source_worktree}/paper/results"
cp -R "${generated_output}/figures" "${source_worktree}/paper/figures"

expansion_manifest="${repo_root}/paper/expansion/results/manifest.json"
if [[ ! -f "${expansion_manifest}" ]]; then
  printf '%s\n' "reviewed expansion manifest not found: ${expansion_manifest}" >&2
  exit 1
fi
# Copy first, then verify the exact expansion tree that the PDFs will consume.
# This closes the gap between checking mutable working-tree artifacts and use.
rm -rf "${source_worktree}/paper/expansion/results" \
       "${source_worktree}/paper/expansion/figures"
cp -R "${repo_root}/paper/expansion/results" \
  "${source_worktree}/paper/expansion/results"
cp -R "${repo_root}/paper/expansion/figures" \
  "${source_worktree}/paper/expansion/figures"
PYTHONPATH="${source_worktree}" "${python_bin}" - \
  "${source_worktree}/paper/expansion" <<'PY'
import sys

from scripts.run_paper_experiments import verify_expansion_artifacts

verify_expansion_artifacts(sys.argv[1])
PY
expansion_source_commit="$(
  "${python_bin}" -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["git"]["source_commit"])' \
    "${source_worktree}/paper/expansion/results/manifest.json"
)"
if [[ "${expansion_source_commit}" != "${source_commit}" ]]; then
  printf '%s\n' \
    "expansion artifacts do not match the paper source commit" \
    "paper source:     ${source_commit}" \
    "expansion source: ${expansion_source_commit}" >&2
  exit 1
fi

(
  cd "${source_worktree}"
  PYTHON_BIN="${python_bin}" \
  PAPER_SOURCE_COMMIT="${source_commit}" \
  SOURCE_DATE_EPOCH="${source_date_epoch}" \
  TECTONIC_EXPECTED_VERSION=0.16.9 \
    scripts/build_paper.sh
)

# Install the complete validated release or restore every previous destination.
publish_bundle \
  "${source_worktree}/paper/results" \
  "${repo_root}/paper/results" \
  "${source_worktree}/paper/figures" \
  "${repo_root}/paper/figures" \
  "${source_worktree}/paper/quantcortex_audit_neurips2026.pdf" \
  "${repo_root}/paper/quantcortex_audit_neurips2026.pdf" \
  "${source_worktree}/paper/quantcortex_audit_neurips2026.sha256" \
  "${repo_root}/paper/quantcortex_audit_neurips2026.sha256" \
  "${source_worktree}/paper/quantcortex_audit_neurips2026.sources.sha256" \
  "${repo_root}/paper/quantcortex_audit_neurips2026.sources.sha256" \
  "${source_worktree}/paper/quantcortex_audit_anonymous.pdf" \
  "${repo_root}/paper/quantcortex_audit_anonymous.pdf" \
  "${source_worktree}/paper/quantcortex_audit_anonymous.sha256" \
  "${repo_root}/paper/quantcortex_audit_anonymous.sha256" \
  "${performance_output}/img" \
  "${repo_root}/docs/img" \
  "${source_worktree}/paper/quantcortex_audit_neurips2026.pdf" \
  "${repo_root}/output/pdf/quantcortex_audit_neurips2026.pdf" \
  "${source_worktree}/paper/quantcortex_audit_anonymous.pdf" \
  "${repo_root}/output/pdf/quantcortex_audit_anonymous.pdf" \
  "${source_worktree}/output/pdf/quantcortex_tae_camera_ready.pdf" \
  "${repo_root}/output/pdf/quantcortex_tae_camera_ready.pdf" \
  "${source_worktree}/paper/build_manifest.json" \
  "${repo_root}/paper/build_manifest.json"

printf '%s\n' "released paper artifacts from source commit ${source_commit}"
printf '%s\n' "manifest timestamp: ${generated_at}"
