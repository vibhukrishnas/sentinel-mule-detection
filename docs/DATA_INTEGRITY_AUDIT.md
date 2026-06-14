# SENTINEL — Data Integrity Audit

*Automated leakage scan · base fraud rate 0.8919% · 598 flagged features*

A leakage feature is one that encodes the OUTCOME rather than pre-event behaviour. Training on it produces a fake ~1.0 score that collapses in production. This audit scores every feature on four leak signatures.

## Severity summary
| Severity | Meaning | Count |
|---|---|---|
| CRITICAL | label-proxy / ≥90%-pure bucket / AUC≥0.98 | 1 |
| HIGH | ≥50%-pure bucket / AUC≥0.95 | 6 |
| MEDIUM | 20–50%-pure bucket (review) | 52 |
| WATCH | 10–20%-pure bucket | 539 |

**Recommended action:** auto-exclude the 7 CRITICAL/HIGH non-bank-listed features before modelling (SENTINEL does this automatically).

## Top flagged features
| Feature | Severity | Univ. AUC | Best bucket fraud-rate | Lift vs base | Bank-listed |
|---|---|---|---|---|---|
| F2230 | CRITICAL | 0.593 | 100% | 112.1× | no |
| F3484 | HIGH | 0.592 | 82% | 91.7× | no |
| F3700 | HIGH | 0.534 | 80% | 89.7× | no |
| F3706 | HIGH | 0.568 | 55% | 61.2× | no |
| F3712 | HIGH | 0.568 | 55% | 61.2× | no |
| F3490 | HIGH | 0.547 | 50% | 56.1× | no |
| F3496 | HIGH | 0.547 | 50% | 56.1× | no |