# Paper reproduction records

Use this guide to find the implementation records, commands, and exact
identifiers for reproducing or inspecting the manuscript's results. Start with the
[paper build and coordinated release instructions](../paper/README.md).

## Inputs and experiment records

| Record | Location | What it contains |
|---|---|---|
| Retrospective experiment | [`paper/results/manifest.json`](../paper/results/manifest.json) | Source revision, data description, input digest, environment, accounting settings and output hashes |
| Expansion experiment | [`paper/expansion/results/manifest.json`](../paper/expansion/results/manifest.json) | The corresponding records for both ETF sets and all strategy families |
| Expansion design | [`paper/preregistration.md`](../paper/preregistration.md) and [`protocol.json`](../paper/expansion/protocol.json) | Rules, model settings, seeds, comparisons and uncertainty procedure fixed in the local design record |
| Retrospective evaluation contract | [`evaluation_contract.json`](../paper/results/evaluation_contract.json) | Machine-readable declarations for the recorded evaluation |
| Target identity | [`retrospective target hashes`](../paper/results/target_tape_hashes.json) and [`expansion target hashes`](../paper/expansion/results/target_tape_hashes.json) | Fingerprints of the empirical allocation records |
| Data provenance | [`expansion data record`](../paper/expansion/results/data_provenance.json) and the retrospective manifest | Provider, adjustment field, retrieval details and input identity |
| PDF provenance | [`build_manifest.json`](../paper/build_manifest.json) and [`source checksums`](../paper/quantcortex_audit_neurips2026.sources.sha256) | Exact paper source, compiler and artifact identities |
| Data publication review | [`data-source-due-diligence.md`](data-source-due-diligence.md) | Repository data-acceptance rules and the evidence needed for publication rights |

The historical paths require three local price matrices: the retrospective
matrix of six risky ETFs plus SHV at `local_data/published_rotation_prices.csv`
and the two expansion matrices under `local_data/expansion/`. Reproduction also
requires their source records and the recorded numerical environment. The raw
matrices remain local. Public payload hashes provide fingerprints;
reconstructing target paths requires the matrices, generation code, and recorded
environment. A fingerprint identifies a matching copy when one is available.
Public synthetic examples exercise the accounting without those inputs.

The design filename records the repository's local terminology. The authors
controlled the design record and its timestamps. External registration and an
independent chronology were not established. Both studies examine historical
periods already available during design.

## Validation and failure behavior

The current machine interface uses version 1. It rejects unsupported versions
and unknown fields. Complete decision books must have valid timestamps,
unique symbol records, finite weights, the required long-only bounds and gross
exposure within the declared limit. Empty record collections, incomplete books,
and duplicate records fail validation.

The generator owns feature and training-label availability. Record validation
checks records and weights; execution alignment is a separate evaluator check.
A target with no later execution bar remains unexecuted. Residual cash requires
a complete aligned return series, and the engines require a transaction-cost
model. A missing cash observation stops the run instead of receiving zero.

The order ledger records intent before submission and rejects stale state
updates. An unresolved intent stops uncertain retries and new identifiers for
the same symbol. The tests exercise local persistence and simulated failures.
Authenticated transport, venue behavior, pending account exposure and delivery
across concurrent clients require separate operational evidence.

## Cost-label convention

Some historical metadata uses a label that can suggest a one-way cost. The
implementation charges `0.0013 * (B + S)`, where `B` and `S` are purchases and
sales as fractions of pre-trade NAV. One-way turnover is `max(B, S)` and is a
different quantity. Retain the original metadata when examining an old release
and use the recorded formula to interpret its charge. The paper states the
cost base directly.

## Figures and displayed results

The current manuscript has 13 figures: six retrospective figures and seven
expansion figures.

[`scripts/run_paper_experiments.py`](../scripts/run_paper_experiments.py)
generates the retrospective figures. Figure 1 originates in
[`paper/audit_protocol.drawio`](../paper/audit_protocol.drawio); its adjacent
embedded PNG and PDF exports are reviewed inputs to the release. The
[paper README](../paper/README.md#figure-1-source) gives the export commands.

[`scripts/run_expansion_experiments.py`](../scripts/run_expansion_experiments.py)
generates seven expansion figures and the rank comparison table.
The table uses `family_summary.csv` and `rank_reversals.csv` under
`paper/expansion/results/`; its body is generated in `generated_values.tex`.
For each ETF set, four strategy rows give sample Sharpe relative to SHV and
rank under baseline accounting, infeasible retroactive assignment, and zero
modeled cost. Rank one is highest. Gradient boosting Sharpe entries are the mean
of the five individual seed Sharpes. Three decimal places retain the small
sector zero-cost gap.
Each timing or cost comparison changes four family positions through two
pairwise swaps across the two ETF sets. The table replaces the former rank
figure using the same results and comparison definitions. Tables and prose use
Gradient boosting; compact plot labels use GBRT. Appendix Table 7 reports counts
below zero, including zero, and above zero for block lengths of 5, 21, and 63
sessions.

Two descriptive analyses added during revision also reuse the recorded paths.
`yearly_performance.csv` contains 128 run-year rows of annualized arithmetic
excess return over SHV. Calendar-year slicing preserves holdings and costs
across year boundaries. `seed_contract_effects.csv` contains 50 paired Sharpe
differences, covering all five comparisons for each of the five gradient boosting
seeds in both ETF sets. Their figures show annual variation and variation across
seeds. They add no new model fit, target path, data request, or hypothesis test
and do not change the frozen primary protocol.

The convention forest plots group rows by comparison. Their family colors and
shapes stay consistent across return and Sharpe. Hollow markers indicate intervals
including zero; this classification uses the unrounded interval endpoints.
The full point estimates and bounds remain in `contract_effects.csv`, including
intervals too narrow to resolve at the figure's shared scale.

The engine figure uses all 16 rows in `engine_conformance.csv`: one path for each
rule and five paths for the gradient boosting model in each ETF set. It shows daily
absolute maxima and signed terminal wealth differences separately. These
quantities also differ from the retrospective figure's cumulative arithmetic
sum of daily return differences. The seed figure compares individual Sharpes
with the rules, and the seed-effect matrix adds a detail view of its small
engine differences. All values come from the existing aggregate records.

Plot generators share the custom palette in
[`quantcortex/backtest/metrics/plotting.py`](../quantcortex/backtest/metrics/plotting.py).
Figure captions define aggregation, comparison signs, units and
intervals. Regenerate figures through the release process so the manifests
continue to bind the displayed output to its source.

## Terminology and review

Future prose edits should preserve these distinctions and literal field names:

- Retain **circular block bootstrap**: sample blocks of consecutive trading
  dates with replacement, wrapping from the last date to the first. Shared
  sampled indices keep compared returns, cash and seed pairs aligned. The
  method follows [Politis and Romano](https://statistics.stanford.edu/technical-reports/circular-block-resampling-procedure-stationary-data);
  the frozen protocol specifies this study's intervals and scope.
- Compute **Sharpe relative to SHV** from daily portfolio returns minus SHV
  returns: their mean divided by their sample standard deviation, multiplied
  by the square root of 252. Both moments use differential returns,
  as in [Sharpe's definition](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm).
- **Pseudo-shares** are accounting units valued at adjusted closing prices,
  not broker share quantities. [Yahoo's definition](https://in.help.yahoo.com/kb/adjusted-close-sln28256.html)
  includes adjustments for splits and dividend distributions; those prices do
  not establish executable fills.
- Use **subplot** for a graphical subdivision; retain **panel** for a data
  collection, such as an ETF price panel, and in literal identifiers.
- The four strategy colors use the **Okabe-Ito palette**; the extended
  plotting palette adds other colors. Markers and line styles distinguish
  quantities where needed, following [Okabe and Ito's guidance](https://jfly.uni-koeln.de/color/index.html).
  These choices do not establish accessibility of every rendered figure.

## Historical compute record

[`paper/COMPUTE.md`](../paper/COMPUTE.md) records a historical timing experiment
at source commit `e0443b8f77cd23aee8f1fa64a2bc237e47626c47`, with fixed generation
timestamp `2026-06-18T23:53:33Z`. These measurements concern an earlier version
of the experiments. The current source revision is recorded in the release
manifests.

The Apple M1 host had 8 GiB of system memory. The expansion wrapper took
325.19 seconds and the retrospective analysis and paper wrapper took
145.83 seconds, totaling 471.02 seconds. Commands, CPU time, environment and
excluded activities are in the compute record. Host capacity is not a peak
memory measurement. Those historical timings also exclude incremental checking
cost and total project compute.

## Release and submission

Use the approved source revision and existing authorized inputs. The release
order is expansion first, then paper, from the same clean commit. Changed
source or inputs require fixed UTC timestamps for both wrappers and an absolute
Python path. The paper README is authoritative for these commands and for the
checks required before replaying an unchanged build. An input refresh requires
separate authorization.

The current accepted-paper requirements are on the
[official TAE site](https://tai-eval.github.io/). The identified camera-ready
uses the unmodified NeurIPS 2026 style with `dblblindworkshop,final`, named
authors and the required workshop title. The main-text limit is ten pages,
including figures and tables. The upload deadline is October 31, 2026 AoE.
Submit through the paper's Camera Ready Revision button on OpenReview after
completing the author-side checks.

The [official accepted-paper list](https://tai-eval.github.io/papers/)
confirms Submission 423 and links to
[its OpenReview forum](https://openreview.net/forum?id=Yjm9wkbzMb).
The repository does not establish the identity of the currently uploaded PDF.
Author profiles, reviewer-response verification, final upload, presenter
arrangements and provider publication rights remain separate tasks.
