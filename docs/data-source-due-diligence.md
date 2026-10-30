# Research data

Before collecting data for a study, record the source and the terms governing
its use:

- provider and dataset
- contracting party
- permitted uses
- redistribution terms
- reviewer-access path
- retrieval timestamp
- adjustment method
- symbol mapping
- content digest

## Published report example

The historical report uses Yahoo Finance adjusted closing prices retrieved
through yfinance 1.5.1 on August 29, 2026 with owner authorization. The raw matrix
is kept out of Git. The public charts and their
[manifest](img/performance_manifest.json) record the source, dates, configuration,
software environment, metrics, and output hashes. Provider authorization for
public publication has not been independently verified.

The retained input SHA-256 is
`9c499ec55c630b596aa724a78de36ee5d98a2edf453b534080e1ad105b46a9c4`.
A matching digest identifies an input copy; it cannot reconstruct the prices or
grant permission to use them. Exact reproduction depends on access to that matrix
and the recorded numerical environment. Synthetic tests can check accounting and
execution behavior without those observations.

## New studies

Document these conditions before inspecting results:

1. Lawful research use and publication of derived results are permitted.
2. Adjustment, calendar, timestamp, and survivorship policies are explicit.
3. Reviewer access is described honestly; redistribution and access are
   separate questions.
4. A frozen local input is content-addressed and retained under the applicable
   terms.
5. Reproducibility, data permissions, and existing-asset terms are assessed
   individually rather than inferred from a generic license label.

Public-domain macro or rates series can support an open real-data panel, but
constructed asset returns introduce modeling assumptions and cannot substitute
for quoted equity prices without explicit justification. Paid commercial data
may permit research use while still forbidding redistribution. Synthetic data
remains appropriate for contract and regression tests only, never headline
performance.

## Source records

For each candidate source, create a dated record under ignored local research
notes before acquisition. Do not add provider files to Git. If the permission
basis is ambiguous, disclose that limitation and do not claim the panel is open,
independently reproducible, or submission-ready.
