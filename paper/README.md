# Research Paper

`main.tex` is the identified NeurIPS 2026 TAE manuscript on version-1
evaluation contracts and executable checks for portfolio backtests. It presents
an exact return decomposition, a retrospective allocation shortfall, and an
expansion across four strategy families and two ETF sets under a recorded design.
The expansion compares timing, cash, and cost conventions, a comparison portfolio
after costs, and two engines, with paired block resampling: each draw uses the same sampled
date blocks across the compared return paths. The timing, cash, and cost
comparisons reuse the same sequence of decision times and target weights, input
data, and primary event engine while changing one convention.
Every specified gradient boosting seed is retained.
The current manuscript includes 13 figures: six retrospective figures and seven
expansion figures.

Two descriptive appendix figures added during revision show all eight calendar
years of arithmetic excess return over SHV and all five convention effects for
each gradient boosting seed. These reuse the original continuous baseline paths
and the six recorded variants. They extend the presentation after inspection of the
aggregate results; they do not change the frozen primary protocol.
The derived tables are `expansion/results/yearly_performance.csv` with 128
run-year rows and `expansion/results/seed_contract_effects.csv` with 50 paired
Sharpe differences. Annual slicing preserves holdings and costs across year
boundaries. Gradient boosting cells in the annual heatmap average individual
metrics. A supporting table reports mean risky exposure, annual gross traded notional, and the annualized
cost deduction recorded by the baseline event engine.

Appendix A gives a worked timing example over four synthetic closes, with zero
trading charges and cash returns. It shows how retroactive assignment can raise
or lower reported returns.

The [reproduction records](../docs/paper-reproducibility.md) locate the input,
target, environment and artifact records, explain the historical cost label,
and identify the compute experiment. Use those records and the manifests for
exact hashes and operational file paths.
`anonymous.tex` suppresses author, repository, system-name, and exact fingerprint
identifiers while retaining the `dblblindworkshop,final` package options. `submission.tex`
builds the identified TAE camera-ready with
`\usepackage[dblblindworkshop,final]{neurips_2026}` and the long body. The
work appears at the non-archival TAE workshop. The
[official accepted-paper list](https://tai-eval.github.io/papers/) confirms
Submission 423 and both authors. The repository does not establish the identity
of the PDF currently uploaded to OpenReview.

The style file and checklist were obtained from the official
[NeurIPS 2026 formatting package](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip)
without modification to the style implementation or checklist questions.
The identified camera-ready and arXiv versions omit the checklist. The anonymous
review copy retains it, and `checklist.tex` remains available in the repository.
The identified final versions also omit the acknowledgments section at the
authors' request. References begin on a new page after the conclusion.
The appendices begin on a new page after the references.
The anonymous copy retains its existing acknowledgments.
The retained [answered checklist](checklist.tex) records all 16 answers,
including the limits on data access, compute reporting, provider permissions,
and the scope of LLM assistance. [Data-source due diligence](../docs/data-source-due-diligence.md)
distinguishes owner authorization from independently verified provider terms.

## Rebuilding an unchanged release

`scripts/build_paper.sh` overwrites tracked PDFs, checksums, and build provenance.
A bare invocation is not an unchanged release replay. From the repository root,
first verify all entries with
`sha256sum --check paper/quantcortex_audit_neurips2026.sources.sha256`.
Only if that check passes, set `PAPER_SOURCE_COMMIT` and `SOURCE_DATE_EPOCH` from
`source_commit` and `source_date_epoch` in the recorded
`paper/build_manifest.json` before invoking the build. If any bound source has
changed, use the [coordinated release](#coordinated-release) instead; do not label
changed sources with old release metadata.

The verified public PDF is written to
`output/pdf/quantcortex_audit_neurips2026.pdf` and copied to the tracked
`paper/quantcortex_audit_neurips2026.pdf` publication artifact. The tracked
anonymous build is `paper/quantcortex_audit_anonymous.pdf`; the untracked TAE
camera-ready PDF is `output/pdf/quantcortex_tae_camera_ready.pdf`. Tectonic 0.16.9 is
required, and the build fails on another version. The build requires Poppler
inspection tools plus `strings` and `flock`. It checks the ten-page body limit
for the public, TAE camera-ready, and anonymous derivative. It also checks
text-rectangle overflow, font embedding, anonymous-PDF text, camera-ready author
and workshop-title text, metadata, URLs, embedded files, raw strings, and exact
fingerprints. Visual QA remains a separate required step. The build updates both
tracked PDF checksum files and `build_manifest.json`.
The anonymous derivative shares the accepted paper's content and page limit;
it is not an initial-submission version under the separate eight-page limit.
The page check treats the official acknowledgment heading or References as
the end of the main text. The official NeurIPS formatting instructions exclude
acknowledgments from the content-page count. A heading on page 11 must be its
first substantive line, so main text spilling onto that page still fails.
The build also writes `quantcortex_audit_neurips2026.sources.sha256`, which binds the
tracked PDF to the snapshotted LaTeX, bibliography, generated values, figures,
build script, and bundle publisher that produced it. Both manifest files use
repository-root-relative paths. Tectonic reads only the snapshot; the build also
fails before publication if live bound source differs from that snapshot after
compilation. A shared lock serializes direct builds and both release wrappers.
Historical host details and measured wrapper wall times are in
[`COMPUTE.md`](COMPUTE.md).

## Table emphasis

Table headings spell out metric names, including compound annual growth rate,
annualized arithmetic return, and maximum drawdown. Longer headings wrap at
word boundaries. Strategy names stay on one line in the rank, baseline
performance, and trading tables at the existing font size. Tables and prose use
Gradient boosting; compact plot labels use GBRT, defined in the target-generator
table. The SHV ticker remains unchanged. Numeric precision,
units, comparison directions, and seed aggregation are part of the meaning
and must survive wording changes. Results paragraphs explain the comparisons
and retain numerical examples when they clarify a mechanism or close ranking.
Detailed values remain in the tables, plots, and aggregate CSVs.

The rank table highlights the highest sample Sharpe within each ETF set and
convention. The baseline performance table highlights the highest return,
Sharpe, and CAGR, and the shallowest drawdown within each set. The generator
uses unrounded baseline values and the rank record's 12-significant-digit
precision for Sharpe ordering, highlighting all maxima at the respective
precision. Bold marks sample extrema; it does not establish statistical
superiority or make retroactive assignment feasible. Accounting components,
interval counts, and trading summaries have no common best-value rule and
receive no score emphasis. Appendix Table 7 reports counts below zero, including
zero, and above zero for block lengths of 5, 21, and 63 sessions. Gradient
boosting Sharpe entries are means of the individual seed Sharpes.

## Figure conventions

The convention plots group the four families under each comparison and retain
one scale per metric across both ETF sets. Family colors and shapes identify
the rows. Their blue, vermilion, bluish green, and reddish purple follow the
[Okabe-Ito palette](https://jfly.uni-koeln.de/color/). Shapes and line styles
retain distinctions in grayscale; text stays dark. The signed heatmaps retain
their separate red-to-blue scale centered at zero.
Hollow markers indicate intervals that include zero; filled markers
indicate intervals that exclude zero. This uses unrounded bounds, including an
endpoint exactly at zero. Narrow intervals can lie behind the markers, so use
`expansion/results/contract_effects.csv` for exact estimates and endpoints.
The marks describe pointwise intervals, not simultaneous coverage or economic
importance.

The retrospective interval plot uses the same basket return before costs for
both Gross and Net. Net deducts the strategy's trading costs. The engine
comparison footer summarizes cumulative arithmetic return differences, which
differ from the gap between the compounded wealth paths.

The seed comparison includes the three rule baselines. The engine diagnostic
shows each rule and each gradient boosting seed separately, with maximum
absolute daily return differences and signed terminal wealth differences on distinct axes.
The seed-effect matrix retains its common color scale and adds an engine detail
with a separately labeled scale. These displays reuse existing aggregate
results; no fitted path or statistical procedure changes.

## Figure typography

Paper plots and the native diagram use DejaVu Serif, with STIX for plot
mathematics. Explicit selection keeps the same face on
systems with or without Times New Roman. Matplotlib exports retain embedded TrueType fonts
with `pdf.fonttype=42`; report and notebook plots retain their existing style.

Judge text size at the final manuscript width, using the saved PDF crop box
rather than the nominal canvas alone. Paper plots aim for approximately
9-point subplot titles, 8-point axis labels, and 7 to 7.5-point ticks, legends,
and annotations after placement. Keep every figure's text below the main
text size of 10 TeX points, equivalent to about 9.96 PDF points. Measure the
embedded text after manuscript scaling; a tightly cropped export can enlarge
text even when the nominal canvas is wider than its placement. Both interval
plots use the full text width.
Review all labels, subplots, legends, and captions in the compiled paper after
changing a canvas, font, or inclusion width.

## Figure 1 source

`audit_protocol.drawio` is the editable source for Figure 1. The adjacent
`audit_protocol.drawio.pdf` and `audit_protocol.drawio.png` are reviewed exports
from Draw.io Desktop 31.3.2 with the diagram embedded. Dashed containers group
Generation, Portfolio simulation, and Evaluation. A shared market-input band
sits above these stages. Its branches supply price history to generation,
execution prices to rebalancing, prices and cash returns to portfolio valuation,
and SHV to the evaluation benchmark. Generation applies the decision-time and
label-availability restrictions shown in its box. Portfolio simulation contains
trade costs, rebalancing, and portfolio state; the caption identifies the
baseline event engine. The feedback arrow carries holdings and cash into the
next rebalance. Allocation records, rebalancing, portfolio state, and daily records
share a horizontal path. The allocation record has a separate compartment for its
validation conditions. Daily records feed performance measures and resampling
separately. The main transfers use slightly heavier arrows than supporting
inputs; the state feedback runs beneath them.
The stage title bands are pale blue for generation, green for portfolio
simulation, and orange for evaluation. Native divided boxes draw
each fill, outer border, and title separator together. Keep the rule below each
title visible; separate opaque header rectangles can cover the outer border.
White body areas, small editable line icons, and dark text keep the operations
readable. Labels and arrows carry the meaning independently of color; shared
inputs and record compartments remain
neutral. Icons use native Draw.io shapes, with no external image or symbol-font
dependencies. Preserve the text size when changing spacing.
Refresh both exports after editing:

```bash
xvfb-run -a drawio -x -f pdf -e --crop -b 10 \
  -o paper/audit_protocol.drawio.pdf paper/audit_protocol.drawio --disable-gpu
xvfb-run -a drawio -x -f png -e --width 2000 -b 10 \
  -o paper/audit_protocol.drawio.png paper/audit_protocol.drawio --disable-gpu
```

Repair the embedded PNG with the Drawio skill's `scripts/repair_png.py` after
export; this desktop version can omit its final IEND bytes. Check that both
exports retain the source XML and inspect them at the paper's 5.5-inch width in
color and grayscale. Keep DejaVu Serif and at least 7-point printed labels.
The experiment generator copies these
reviewed inputs into `figures/`; it does not invoke Draw.io during a release.
The source fingerprint covers the XML and both exports. This preserves the
approved rendering without adding a desktop dependency to experiment generation.

## Coordinated release

Use only authorized local adjusted-close matrices. Source commits require
explicit user approval. After approval, commit release-critical source, then run
the expansion wrapper first and the paper wrapper second from that same source
commit, with no intervening commit.

For a new public export, start from a vetted source root P without private Git
history. Run the two wrappers below from P with fixed UTC timestamps and the
already authorized local inputs, then commit generated outputs in artifact
child A. Both manifests must name P; their ancestry and source-byte checks must
pass in A. Imported manifests naming unavailable private source commits are not
an unchanged public replay. Existing owner authorization for derived publication
is retained; this rebuild does not authorize a new provider request.

The historical protocol-freeze identifier
`4018f4063f46889f41d6981db5a71079e1dbd713` refers to private development history.
Retain its protocol bytes and hash as recorded evidence, but do not describe the
identifier as publicly reviewable history or proof of pre-data publication.
The new P-to-A lineage establishes the source of the regenerated artifacts,
not the chronology of the original study. Workflow definitions and private
validation receipts do not establish current public CI results.

Changed release-critical source or inputs require explicit, fixed UTC timestamps
for both wrappers, even if only paper source changed. Replace the placeholders
below with those timestamps. From the repository root, set an absolute
`PYTHON_BIN` once so the detached worktrees use the reviewed environment:

```bash
export PYTHON_BIN="$PWD/.venv/bin/python"
QUANTCORTEX_EXPANSION_GENERATED_AT="YYYY-MM-DDTHH:MM:SSZ" \
  scripts/release_expansion_artifacts.sh local_data/expansion &&
QUANTCORTEX_GENERATED_AT="YYYY-MM-DDTHH:MM:SSZ" \
  scripts/release_paper_artifacts.sh local_data/published_rotation_prices.csv
```

This example retains reviewed inputs and provenance; leave input-refresh digest
and provenance overrides unset. Only for unchanged reviewed source and inputs,
omit both generation-timestamp overrides to reuse the recorded source commits and
timestamps. A source-only release is not an input refresh or permission to
refetch data or publish results.

The retrospective input must contain `QQQ`, `VGT`, `GLD`, `TLT`, `SPY`, `VIG`, and
`SHV` and must match the SHA-256 digest in `results/manifest.json`; by default the
paper wrapper rejects another matrix rather than attaching reviewed provenance
to it. Only for a separately authorized new input, also set
`QUANTCORTEX_EXPECTED_INPUT_SHA256` and all four provenance variables:
`QUANTCORTEX_DATA_PROVIDER`, `QUANTCORTEX_PERMISSION_BASIS`,
`QUANTCORTEX_RETRIEVED_AT`, and `QUANTCORTEX_ADJUSTMENT_METHOD`.
The paper wrapper requires committed tracked source, regenerates the reviewed
`docs/img/` gallery and paper experiment in a detached clean worktree with the
input mounted outside it, verifies the recorded source commit and clean start
state, then publishes the validated bundle with rollback on partial failure.
Before staging, the publisher rejects destination aliases and combinations of
a directory and its contents. It resolves parent directories while preserving
replacement of a final symlink itself.
The fixed experiment requires complete rows, performs no forward fill, and rejects
fewer than 274 pre-evaluation sessions. Raw provider data is not committed.
Aggregate tables, generated LaTeX values, figures, the explicit experiment
design, package source-tree fingerprints, dependency lock, source metadata,
configuration hash, clean source revision, package and thread-library versions,
and artifact hashes are committed in the manifest. `return_decomposition.csv` records the exact
allocation, exposure-timing, passive-exposure, cost, and net-cash components.
`ablation_uncertainty.csv` records joint 21-session block-bootstrap intervals
for every named overlay variant.
`protocol_switches.csv` retains its historical filename and records the audited
result beside baseline-conditional diagnostics. The primary 21-session joint
block bootstrap is accompanied by 5- and 63-session sensitivity results.
`sharpe_uncertainty.csv` directly resamples the conventional sample Sharpe
statistic. `comparator_diagnostics.csv` records the comparison portfolio following
the strategy's target exposure at the next bar, after its own trading costs.
It remains conditional on that target path. `evaluation_contract.json` records
the machine-readable semantics.
`target_tape_hashes.json` records the canonical payload hash, decision count,
record count, and symbol set for every audited strategy variant without
publishing the underlying provider matrix.

The expansion input directory must contain `us_sector_etfs.csv`,
`country_equity_etfs.csv`, and their `.metadata.json` sidecars. Only for a separately
authorized panel refresh, also set
`QUANTCORTEX_EXPANSION_US_SECTOR_INPUT_SHA256` and
`QUANTCORTEX_EXPANSION_COUNTRY_INPUT_SHA256`; both exact digests are required.
The wrapper rejects a protocol or input hash mismatch, runs in a detached
worktree, and publishes only aggregate CSV/JSON, generated LaTeX, and figures
under `paper/expansion/`. The paper wrapper requires these artifacts to name the same
clean source commit.

The primary accounting path is the event-driven engine. It holds fixed quantities
of pseudo-shares between rebalances. These are accounting units valued at
adjusted closing prices. It sizes targets against post-cost NAV and reports
both one-way turnover and gross two-sided traded notional. The
vectorized engine remains an approximation and model-convention sensitivity
diagnostic; equality with pseudo-share accounting is not expected.

Citation keys are checked against `references.bib`, and DOI/arXiv identifiers
are kept explicit. Revalidate the unversioned 2026 preprints before any future
submission because their metadata and claims may change.

The three non-redistributed raw matrices are a material reproducibility limitation. Open
schemas, hashes of empirical allocation records, and synthetic conformance
fixtures reproduce software semantics, not historical returns. The empirical
allocation records are not published, and a payload hash cannot reconstruct them. Exact independent
reproduction requires matching matrices, source and configuration, environment,
and hashes of reconstructed allocation records. Data-source acceptance rules are in
`docs/data-source-due-diligence.md`. For the expansion, the recorded protocol
commit timestamp precedes the recorded provider-request timestamps, but these
records are author-controlled. The protocol was not externally registered.

## Submission Readiness

The active workshop target is the non-archival NeurIPS 2026 TAE workshop.
The live call, checked on 2026-10-08 at
<https://tai-eval.github.io/cfp/>, allows 8 body pages for initial submissions
and 10 content pages for accepted camera-ready papers, excluding references
and appendices. It specifies
`\usepackage[dblblindworkshop, final]{neurips_2026}` with
`\workshoptitle{TAE (Trust-AI-Eval): Can We Trust AI Evaluation?}` for the
camera-ready. The [camera-ready instructions](https://tai-eval.github.io/)
set the deadline to October 31, 2026 (AoE). Upload the identified final PDF
through the submission's Camera Ready Revision button on OpenReview; the
public paper page updates automatically. Accepted papers are non-archival
and will be presented in person as posters on December 12, 2026, in Sydney.
The camera-ready instructions do not request a checklist. At the authors'
request, the identified final and arXiv versions omit it; the separate
anonymous review copy retains it.
Current artifact identity comes from the manifests and PDF hashes. Internal
submission handoffs are not part of the public export; acceptance alone does not
establish a completed upload or the identity of the uploaded PDF.

The tracked public PDF and the camera-ready PDF both use the identified TAE
final style and name the authors. The anonymous PDF remains a separate build
with its anonymity check unchanged.

The non-redistributed inputs prevent bitwise independent empirical reproduction
from this repository alone. Exact-digest regeneration requires the permitted
local matrices and sidecars; a public checkout does not include them. The
manifests record owner authorization on 2026-08-29 for the live-data run and
publication of derived aggregates. That authorization remains in force, while
provider terms have not been independently verified. This unresolved status
establishes neither provider permission nor a prohibition. Document the
applicable rights basis for the contemplated use, including any OpenReview
release or CC BY 4.0 grant; do not infer additional licensing rights from owner
authorization. The repository does not establish a completed OpenReview upload.

Do not edit generated tables or figures by hand. Change the experiment driver,
run its focused tests, regenerate all outputs, inspect the diffs, then rebuild
and visually review every PDF page.
