#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_root="${repo_root}/tmp/pdfs"
output_dir="${repo_root}/output/pdf"
paper_dir="${repo_root}/paper"
python_bin="${PYTHON_BIN:-python3}"
expected_tectonic_version="${TECTONIC_EXPECTED_VERSION:-0.16.9}"
expected_bundle_sha256="${TECTONIC_BUNDLE_SHA256:-6ffe055852f8faf66c0acbe1a7fb27f87b869a90bad1204f3bf4d9683f597c7c}"
publisher_path="${repo_root}/scripts/publish_artifact_bundle.py"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

publish_bundle() {
  "${python_bin}" "${publisher_path}" "$@"
}

verify_hash() {
  local file_path="$1"
  local expected_hash="$2"
  local label="$3"
  local actual_hash
  actual_hash="$(sha256_file "${file_path}")"
  if [[ "${actual_hash}" != "${expected_hash}" ]]; then
    printf '%s\n' "published ${label} digest mismatch" >&2
    exit 1
  fi
}

write_source_manifest() {
  local manifest_path="$1"
  local source_root="$2"
  local relative_path
  local source_hash
  : > "${manifest_path}"
  for relative_path in "${source_files[@]}"; do
    source_hash="$(sha256_file "${source_root}/${relative_path}")"
    printf '%s  %s\n' "${source_hash}" "${relative_path}" \
      >> "${manifest_path}"
  done
}

command -v tectonic >/dev/null 2>&1 || {
  printf '%s\n' "tectonic is required to build the paper" >&2
  exit 1
}
command -v "${python_bin}" >/dev/null 2>&1 || {
  printf '%s\n' "${python_bin} is required to write the build manifest" >&2
  exit 1
}
command -v pdftotext >/dev/null 2>&1 || {
  printf '%s\n' "pdftotext is required for paper validation" >&2
  exit 1
}
command -v pdfinfo >/dev/null 2>&1 || {
  printf '%s\n' "pdfinfo is required for paper validation" >&2
  exit 1
}
command -v pdfdetach >/dev/null 2>&1 || {
  printf '%s\n' "pdfdetach is required for paper validation" >&2
  exit 1
}
command -v pdffonts >/dev/null 2>&1 || {
  printf '%s\n' "pdffonts is required for paper validation" >&2
  exit 1
}
command -v strings >/dev/null 2>&1 || {
  printf '%s\n' "strings is required for paper validation" >&2
  exit 1
}
command -v flock >/dev/null 2>&1 || {
  printf '%s\n' "flock is required for artifact publication" >&2
  exit 1
}

if [[ "${QUANTCORTEX_PUBLICATION_LOCK_HELD:-0}" != "1" ]]; then
  mkdir -p "${repo_root}/tmp"
  exec {publication_lock_fd}>"${repo_root}/tmp/artifact-publication.lock"
  flock -x "${publication_lock_fd}"
  export QUANTCORTEX_PUBLICATION_LOCK_HELD=1
fi

tectonic_version="$(tectonic --version | awk '{print $2}')"
if [[ "${tectonic_version}" != "${expected_tectonic_version}" ]]; then
  printf '%s\n' \
    "expected Tectonic ${expected_tectonic_version}, found ${tectonic_version}" >&2
  exit 1
fi

# Fix PDF timestamps to the experiment vintage. Reproducing the PDF also
# requires unchanged sources, inputs, fonts, Tectonic, and package bundle.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1781568000}"

source_files=(
  "paper/main.tex"
  "paper/anonymous.tex"
  "paper/submission.tex"
  "paper/checklist.tex"
  "paper/references.bib"
  "paper/neurips_2026.sty"
  "paper/preregistration.md"
  "paper/results/generated_values.tex"
  "paper/results/manifest.json"
  "paper/expansion/protocol.json"
  "paper/expansion/results/generated_values.tex"
  "paper/expansion/results/manifest.json"
  "paper/figures/accounting_summary.pdf"
  "paper/figures/audit_protocol.pdf"
  "paper/figures/bootstrap_robustness.pdf"
  "paper/figures/engine_comparison.pdf"
  "paper/figures/return_attribution_and_protocol_switches.pdf"
  "paper/figures/sensitivity_and_ablation.pdf"
  "paper/expansion/figures/baseline_performance.pdf"
  "paper/expansion/figures/contract_effects_return.pdf"
  "paper/expansion/figures/contract_effects_sharpe.pdf"
  "paper/expansion/figures/engine_conformance.pdf"
  "paper/expansion/figures/learned_seed_sensitivity.pdf"
  "paper/expansion/figures/yearly_performance.pdf"
  "paper/expansion/figures/seed_contract_effects.pdf"
  "scripts/build_paper.sh"
  "scripts/publish_artifact_bundle.py"
)

mkdir -p "${build_root}" "${output_dir}"
build_dir="$(mktemp -d "${build_root}/build.XXXXXX")"
cleanup_build() {
  local status=$?
  if [[ "${status}" -eq 0 ]]; then
    rm -rf "${build_dir}"
  else
    printf '%s\n' "failed build retained at ${build_dir}" >&2
  fi
}
trap cleanup_build EXIT
source_snapshot="${build_dir}/source"
prebuild_source_manifest="${build_dir}/prebuild.sources.sha256"
source_manifest="${build_dir}/quantcortex_audit_neurips2026.sources.sha256"
mkdir -p "${source_snapshot}"
for relative_path in "${source_files[@]}"; do
  mkdir -p "${source_snapshot}/$(dirname "${relative_path}")"
  cp "${repo_root}/${relative_path}" "${source_snapshot}/${relative_path}"
done
write_source_manifest "${source_manifest}" "${source_snapshot}"
write_source_manifest "${prebuild_source_manifest}" "${repo_root}"
if ! cmp -s "${source_manifest}" "${prebuild_source_manifest}"; then
  printf '%s\n' "paper sources changed while the build snapshot was created" >&2
  exit 1
fi
publisher_path="${source_snapshot}/scripts/publish_artifact_bundle.py"
(
  cd "${source_snapshot}/paper"
  tectonic --keep-logs --keep-intermediates \
    --outdir "${build_dir}" main.tex
  tectonic --keep-logs --keep-intermediates \
    --outdir "${build_dir}" anonymous.tex
  tectonic --keep-logs --keep-intermediates \
    --outdir "${build_dir}" submission.tex
)
bundle_cache_dir="$(tectonic -X show user-cache-dir 2>/dev/null | tail -n 1)"
bundle_hash_file="$(
  find "${bundle_cache_dir}/hashes" -type f \
    -name '*default_bundle_v33.tar' -print -quit
)"
if [[ -z "${bundle_hash_file}" || ! -f "${bundle_hash_file}" ]]; then
  printf '%s\n' "could not locate the Tectonic default bundle digest" >&2
  exit 1
fi
bundle_sha256="$(tr -d '[:space:]' < "${bundle_hash_file}")"
if [[ "${bundle_sha256}" != "${expected_bundle_sha256}" ]]; then
  printf '%s\n' \
    "expected Tectonic bundle ${expected_bundle_sha256}, found ${bundle_sha256}" >&2
  exit 1
fi
check_body_limit() {
  local pdf_path="$1"
  local label="$2"
  local page_limit="$3"
  local following_page=$((page_limit + 1))
  local body_pages
  local following_page_text
  local first_following_page_text
  # The official NeurIPS template excludes acknowledgment pages from content.
  local boundary_pattern='^[[:space:]]*([0-9]+[[:space:]]+)?(Acknowledgments and Disclosure of Funding|References)[[:space:]]*$'
  body_pages="$(pdftotext -f 1 -l "${page_limit}" -layout "${pdf_path}" -)"
  following_page_text="$({
    pdftotext -f "${following_page}" -l "${following_page}" \
      -layout "${pdf_path}" -
  })"
  if printf '%s\n' "${body_pages}" | grep -Eq \
    "${boundary_pattern}"; then
    return
  elif printf '%s\n' "${following_page_text}" | grep -Eq \
    "${boundary_pattern}"; then
    first_following_page_text="$({
      printf '%s\n' "${following_page_text}" | awk '
        {
          gsub(/\f/, "")
          if ($0 ~ /^[[:space:]]*$/) next
          if ($0 ~ /^[[:space:]]*[0-9]+[[:space:]]*$/) next
          sub(/^[[:space:]]+/, "")
          sub(/[[:space:]]+$/, "")
          sub(/^[0-9]+[[:space:]]+/, "")
          print
          exit
        }
      '
    })"
    if [[ "${first_following_page_text}" != "References" && \
          "${first_following_page_text}" != "Acknowledgments and Disclosure of Funding" ]]; then
      printf '%s\n' \
        "${label} body exceeds its ${page_limit}-page content limit" >&2
      exit 1
    fi
  else
    printf '%s\n' \
      "${label} acknowledgments or references begin after page ${following_page}; body exceeds its ${page_limit}-page content limit" >&2
    exit 1
  fi
}

file_matches() {
  local review_file="$1"
  local label="$2"
  local status
  shift 2
  if grep "$@" "${review_file}"; then
    return 0
  else
    status=$?
  fi
  if [[ "${status}" -eq 1 ]]; then
    return 1
  fi
  printf '%s\n' "${label} validation scanner failed with status ${status}" >&2
  exit 1
}

check_review_anonymity() {
  local pdf_path="$1"
  local label="$2"
  local attachment_count
  local metadata_file="${pdf_path%.pdf}.info.txt"
  local review_file="${pdf_path%.pdf}.review.txt"
  attachment_count="$(pdfdetach -list "${pdf_path}" | awk 'NR == 1 {print $1}')"
  if [[ "${attachment_count}" != "0" ]]; then
    printf '%s\n' "${label} contains embedded files" >&2
    exit 1
  fi
  pdfinfo "${pdf_path}" > "${metadata_file}"
  "${python_bin}" - "${metadata_file}" "${label}" <<'PY'
import sys
from pathlib import Path

metadata = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
label = sys.argv[2]
for field in ("Author", "Creator", "Subject", "Keywords"):
    prefix = f"{field}:"
    for line in metadata:
        if line.startswith(prefix) and line.removeprefix(prefix).strip():
            raise SystemExit(f"{label} has nonempty {field.lower()} metadata")
PY
  {
    pdftotext -layout "${pdf_path}" - &&
      cat "${metadata_file}" &&
      pdfinfo -meta "${pdf_path}" &&
      pdfinfo -url "${pdf_path}" &&
      strings -a "${pdf_path}"
  } > "${review_file}"
  if file_matches "${review_file}" "${label}" -Eqi \
    'Jimmy Chan|jimmybojim|Pepperdine|0009-0002-4670-2828|Kevin Lee|kevinlee69720|University of California|UCLA|0009-0004-0388-9260|micro1|QuantCortex|github\.com/(magnaquant|magnaprog)|/home/kevinlee'; then
    printf '%s\n' "${label} contains an author or repository identifier" >&2
    exit 1
  fi
  if file_matches "${review_file}" "${label}" -Eq \
    '(^|[^[:xdigit:]])[[:xdigit:]]{40}([^[:xdigit:]]|$)|(^|[^[:xdigit:]])[[:xdigit:]]{64}([^[:xdigit:]]|$)'; then
    printf '%s\n' "${label} contains an exact commit or content fingerprint" >&2
    exit 1
  fi
  if file_matches "${review_file}" "${label}" -q $'\357\277\275'; then
    printf '%s\n' "${label} contains a Unicode replacement character" >&2
    exit 1
  fi
}

check_build_log() {
  local log_path="$1"
  local label="$2"
  if file_matches "${log_path}" "${label}" -Eq 'Overfull \\[hv]box'; then
    printf '%s\n' "${label} has content outside the text rectangle" >&2
    exit 1
  fi
}

check_pdf_fonts() {
  local pdf_path="$1"
  local label="$2"
  local font_report="${pdf_path%.pdf}.fonts.txt"
  pdffonts "${pdf_path}" > "${font_report}"
  "${python_bin}" - "${font_report}" "${label}" <<'PY'
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
label = sys.argv[2]
lines = report_path.read_text(encoding="utf-8").splitlines()
if len(lines) < 3:
    raise SystemExit(f"{label} font report is empty")
header = lines[0]
type_start = header.index("type")
encoding_start = header.index("encoding")
embedded_start = header.index("emb")
subset_start = header.index("sub")
for line in lines[2:]:
    font_type = line[type_start:encoding_start].strip()
    embedded = line[embedded_start:subset_start].strip()
    if font_type == "Type 3" or embedded != "yes":
        raise SystemExit(f"{label} has a Type 3 or unembedded font: {line}")
PY
}

check_camera_ready_identity() {
  local pdf_path="$1"
  local label="$2"
  local review_file="${pdf_path%.pdf}.camera.txt"
  local author
  pdftotext -layout "${pdf_path}" - > "${review_file}"
  for author in "Jimmy Chan" "Kevin Lee"; do
    if ! file_matches "${review_file}" "${label}" -F -q "${author}"; then
      printf '%s\n' "${label} is missing the identified author ${author}" >&2
      exit 1
    fi
  done
  if ! file_matches "${review_file}" "${label}" -F -q "TAE (Trust-AI-Eval)"; then
    printf '%s\n' "${label} is missing the TAE workshop title" >&2
    exit 1
  fi
  if file_matches "${review_file}" "${label}" -q "Foundation Models for Temporal Systems"; then
    printf '%s\n' "${label} still names the FMTS workshop" >&2
    exit 1
  fi
  if file_matches "${review_file}" "${label}" -q "Anonymous author"; then
    printf '%s\n' "${label} is still anonymous" >&2
    exit 1
  fi
}

check_build_log "${build_dir}/main.log" "public paper"
check_build_log "${build_dir}/anonymous.log" "anonymous paper"
check_build_log "${build_dir}/submission.log" "TAE camera-ready"
check_pdf_fonts "${build_dir}/main.pdf" "public paper"
check_pdf_fonts "${build_dir}/anonymous.pdf" "anonymous paper"
check_pdf_fonts "${build_dir}/submission.pdf" "TAE camera-ready"
# TAE camera-ready versions allow 10 content pages (https://tai-eval.github.io/cfp/).
check_body_limit "${build_dir}/main.pdf" "public paper" 10
# The anonymous derivative retains the accepted paper's full content.
check_body_limit "${build_dir}/anonymous.pdf" "anonymous paper" 10
check_body_limit "${build_dir}/submission.pdf" "TAE camera-ready" 10
check_review_anonymity "${build_dir}/anonymous.pdf" "anonymous paper"
check_camera_ready_identity "${build_dir}/submission.pdf" "TAE camera-ready"

paper_hash="$(sha256_file "${build_dir}/main.pdf")"
anonymous_hash="$(sha256_file "${build_dir}/anonymous.pdf")"
submission_hash="$(sha256_file "${build_dir}/submission.pdf")"

postbuild_source_manifest="${build_dir}/postbuild.sources.sha256"
write_source_manifest "${postbuild_source_manifest}" "${repo_root}"
if ! cmp -s "${prebuild_source_manifest}" "${postbuild_source_manifest}"; then
  printf '%s\n' "paper sources changed during the build" >&2
  exit 1
fi

source_commit="${PAPER_SOURCE_COMMIT:-$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || printf unavailable)}"
source_manifest_hash="$(sha256_file "${source_manifest}")"
build_manifest="${build_dir}/build_manifest.json"
"${python_bin}" - \
  "${build_manifest}" \
  "${source_commit}" \
  "${SOURCE_DATE_EPOCH}" \
  "${tectonic_version}" \
  "${bundle_sha256}" \
  "${paper_hash}" \
  "${anonymous_hash}" \
  "${submission_hash}" \
  "${source_manifest_hash}" <<'PY'
import json
import sys
from pathlib import Path

(
    output_path,
    source_commit,
    source_date_epoch,
    tectonic_version,
    bundle_sha256,
    pdf_sha256,
    anonymous_pdf_sha256,
    submission_pdf_sha256,
    source_manifest_sha256,
) = sys.argv[1:]
manifest = {
    "schema_version": 2,
    "source_commit": source_commit,
    "source_date_epoch": int(source_date_epoch),
    "tectonic_version": tectonic_version,
    "tectonic_bundle": {
        "name": "default_bundle_v33.tar",
        "sha256": bundle_sha256,
    },
    "path_base": "repository_root",
    "pdf": {
        "path": "paper/quantcortex_audit_neurips2026.pdf",
        "sha256": pdf_sha256,
    },
    "anonymous_pdf": {
        "path": "paper/quantcortex_audit_anonymous.pdf",
        "sha256": anonymous_pdf_sha256,
    },
    "submission_pdf": {
        "path": "output/pdf/quantcortex_tae_camera_ready.pdf",
        "sha256": submission_pdf_sha256,
        "tracked": False,
    },
    "source_manifest": {
        "path": "paper/quantcortex_audit_neurips2026.sources.sha256",
        "entry_path_base": "repository_root",
        "sha256": source_manifest_sha256,
    },
}
Path(output_path).write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="ascii",
)
PY

paper_checksum="${build_dir}/quantcortex_audit_neurips2026.sha256"
anonymous_checksum="${build_dir}/quantcortex_audit_anonymous.sha256"
printf '%s  %s\n' "${paper_hash}" "quantcortex_audit_neurips2026.pdf" \
  > "${paper_checksum}"
printf '%s  %s\n' "${anonymous_hash}" "quantcortex_audit_anonymous.pdf" \
  > "${anonymous_checksum}"

# Install the complete validated bundle or restore every previous destination.
publish_bundle \
  "${build_dir}/main.pdf" \
  "${output_dir}/quantcortex_audit_neurips2026.pdf" \
  "${build_dir}/main.pdf" \
  "${paper_dir}/quantcortex_audit_neurips2026.pdf" \
  "${build_dir}/anonymous.pdf" \
  "${output_dir}/quantcortex_audit_anonymous.pdf" \
  "${build_dir}/anonymous.pdf" \
  "${paper_dir}/quantcortex_audit_anonymous.pdf" \
  "${build_dir}/submission.pdf" \
  "${output_dir}/quantcortex_tae_camera_ready.pdf" \
  "${source_manifest}" \
  "${paper_dir}/quantcortex_audit_neurips2026.sources.sha256" \
  "${paper_checksum}" \
  "${paper_dir}/quantcortex_audit_neurips2026.sha256" \
  "${anonymous_checksum}" \
  "${paper_dir}/quantcortex_audit_anonymous.sha256" \
  "${build_manifest}" \
  "${paper_dir}/build_manifest.json"
verify_hash "${paper_dir}/quantcortex_audit_neurips2026.pdf" \
  "${paper_hash}" "public paper"
verify_hash "${paper_dir}/quantcortex_audit_anonymous.pdf" \
  "${anonymous_hash}" "anonymous paper"
verify_hash "${output_dir}/quantcortex_tae_camera_ready.pdf" \
  "${submission_hash}" "TAE camera-ready"

printf '%s\n' \
  "built ${output_dir}/quantcortex_audit_neurips2026.pdf" \
  "built ${output_dir}/quantcortex_audit_anonymous.pdf" \
  "built ${output_dir}/quantcortex_tae_camera_ready.pdf"
