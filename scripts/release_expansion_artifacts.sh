#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"
panel_argument="${1:-${repo_root}/local_data/expansion}"
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
# generated here. Scoping the cleanliness check to those paths keeps the
# wrapper rerunnable while regenerated artifacts sit uncommitted in the
# working tree; the paper wrapper independently verifies those artifacts
# against this release's manifest before republishing them.
release_source_paths=(
  quantcortex
  schemas/canonical_target_tape.schema.json
  pyproject.toml
  poetry.lock
  paper/preregistration.md
  paper/expansion/protocol.json
  scripts/fetch_expansion_data.py
  scripts/publish_artifact_bundle.py
  scripts/release_expansion_artifacts.sh
  scripts/run_expansion_experiments.py
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
    "commit release-critical source changes before releasing expansion artifacts" >&2
  exit 1
fi

panel_dir="$(
  "${python_bin}" -c \
    'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' \
    "${panel_argument}"
)"
if [[ ! -d "${panel_dir}" ]]; then
  printf '%s\n' "expansion panel directory not found: ${panel_dir}" >&2
  exit 1
fi
for panel in us_sector_etfs country_equity_etfs; do
  for suffix in csv metadata.json; do
    if [[ ! -f "${panel_dir}/${panel}.${suffix}" ]]; then
      printf '%s\n' "missing expansion input: ${panel_dir}/${panel}.${suffix}" >&2
      exit 1
    fi
  done
  panel_csv="$(
    "${python_bin}" -c \
      'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' \
      "${panel_dir}/${panel}.csv"
  )"
  case "${panel_csv}" in
    "${repo_root}"/*)
      relative_panel_csv="${panel_csv#${repo_root}/}"
      if git -C "${repo_root}" ls-files --error-unmatch -- \
        "${relative_panel_csv}" >/dev/null 2>&1; then
        printf '%s\n' \
          "reviewed raw expansion input must not be tracked by Git: ${relative_panel_csv}" >&2
        exit 1
      else
        tracking_status=$?
        if [[ "${tracking_status}" -ne 1 ]]; then
          printf '%s\n' \
            "could not determine whether the raw expansion input is tracked" >&2
          exit 1
        fi
      fi
      ;;
  esac
done

current_commit="$(git -C "${repo_root}" rev-parse HEAD)"
reviewed_manifest="${repo_root}/paper/expansion/results/manifest.json"
refresh_sector_digest="${QUANTCORTEX_EXPANSION_US_SECTOR_INPUT_SHA256:-}"
refresh_country_digest="${QUANTCORTEX_EXPANSION_COUNTRY_INPUT_SHA256:-}"
if [[ -n "${refresh_sector_digest}" || -n "${refresh_country_digest}" ]]; then
  if [[ ! "${refresh_sector_digest}" =~ ^[0-9a-f]{64}$ ]] || \
     [[ ! "${refresh_country_digest}" =~ ^[0-9a-f]{64}$ ]]; then
    printf '%s\n' \
      "both explicit expansion refresh digests must be lowercase SHA-256 values" >&2
    exit 1
  fi
  if [[ ! "${QUANTCORTEX_EXPANSION_GENERATED_AT:-}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]]; then
    printf '%s\n' \
      "QUANTCORTEX_EXPANSION_GENERATED_AT must be a fixed UTC timestamp for an input refresh" >&2
    exit 1
  fi
fi

if [[ -n "${QUANTCORTEX_EXPANSION_GENERATED_AT:-}" ]]; then
  source_commit="${current_commit}"
  generated_at="${QUANTCORTEX_EXPANSION_GENERATED_AT}"
elif [[ -f "${reviewed_manifest}" ]]; then
  reviewed_source_commit="$(
    "${python_bin}" -c \
      'import json, sys; print(json.load(open(sys.argv[1]))["git"]["source_commit"])' \
      "${reviewed_manifest}"
  )"
  reviewed_generated_at="$(
    "${python_bin}" -c \
      'import json, sys; print(json.load(open(sys.argv[1]))["generated_at"])' \
      "${reviewed_manifest}"
  )"
  if ! git -C "${repo_root}" cat-file -e "${reviewed_source_commit}^{commit}" \
    >/dev/null 2>&1; then
    printf '%s\n' "reviewed expansion source commit is unavailable" >&2
    exit 1
  fi
  if ! git -C "${repo_root}" diff --quiet \
    "${reviewed_source_commit}" "${current_commit}" -- \
    "${release_source_paths[@]}"; then
    printf '%s\n' \
      "QUANTCORTEX_EXPANSION_GENERATED_AT is required for changed release source" >&2
    exit 1
  fi
  source_commit="${reviewed_source_commit}"
  generated_at="${reviewed_generated_at}"
else
  printf '%s\n' \
    "QUANTCORTEX_EXPANSION_GENERATED_AT is required for the first release" >&2
  exit 1
fi

temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/quantcortex-expansion-release.XXXXXX")"
source_worktree="${temporary_root}/source"
generated_output="${temporary_root}/generated"
input_snapshot="${temporary_root}/inputs"

cleanup() {
  git -C "${repo_root}" worktree remove --force "${source_worktree}" \
    >/dev/null 2>&1 || true
  rm -rf "${temporary_root}"
}
trap cleanup EXIT

git -C "${repo_root}" worktree add --detach "${source_worktree}" "${source_commit}" \
  >/dev/null
publisher_path="${source_worktree}/scripts/publish_artifact_bundle.py"
mkdir -p "${input_snapshot}"
for panel in us_sector_etfs country_equity_etfs; do
  cp "${panel_dir}/${panel}.csv" "${input_snapshot}/${panel}.csv"
  cp "${panel_dir}/${panel}.metadata.json" \
    "${input_snapshot}/${panel}.metadata.json"
done

(
  cd "${source_worktree}"
  MPLCONFIGDIR="${temporary_root}/matplotlib" \
  PYTHONPATH="${source_worktree}" \
  "${python_bin}" scripts/run_expansion_experiments.py \
    --protocol "${source_worktree}/paper/expansion/protocol.json" \
    --panel-dir "${input_snapshot}" \
    --output-dir "${generated_output}" \
    --generated-at "${generated_at}"
)

"${python_bin}" - \
  "${generated_output}/results/manifest.json" \
  "${source_commit}" \
  "${reviewed_manifest}" \
  "${refresh_sector_digest}" \
  "${refresh_country_digest}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected_commit = sys.argv[2]
reviewed_manifest_path = Path(sys.argv[3])
refresh_inputs = {
    "us_sector_etfs": sys.argv[4],
    "country_equity_etfs": sys.argv[5],
}
git = manifest["git"]
if git["source_commit"] != expected_commit:
    raise SystemExit("manifest source commit does not match the release commit")
if git["tracked_worktree_clean_at_start"] is not True:
    raise SystemExit("manifest did not record a clean source worktree")
actual_inputs = {row["panel"]: row["input_sha256"] for row in manifest["data"]}
if all(refresh_inputs.values()):
    if actual_inputs != refresh_inputs:
        raise SystemExit("expansion input matrices do not match explicit refresh digests")
elif reviewed_manifest_path.is_file():
    reviewed = json.loads(reviewed_manifest_path.read_text(encoding="utf-8"))
    expected_inputs = {
        row["panel"]: row["input_sha256"] for row in reviewed["data"]
    }
    if actual_inputs != expected_inputs:
        raise SystemExit("expansion input matrices do not match the reviewed release")
    if manifest["data"] != reviewed["data"]:
        raise SystemExit("expansion data provenance changed from the reviewed release")
root = manifest_path.parents[1]
for relative, expected in manifest["artifacts"].items():
    artifact = root / relative
    actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"artifact digest mismatch: {relative}")
PY

# Install both validated trees or restore both previous destinations.
publish_bundle \
  "${generated_output}/results" \
  "${repo_root}/paper/expansion/results" \
  "${generated_output}/figures" \
  "${repo_root}/paper/expansion/figures"

printf '%s\n' "released expansion artifacts from source commit ${source_commit}"
printf '%s\n' "manifest timestamp: ${generated_at}"
