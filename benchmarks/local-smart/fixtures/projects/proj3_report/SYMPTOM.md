# Project 3: report aggregation

**Fixed fixture for Phase 3 Test A.** Contains one real bug and one misleading clue.

## Symptom

`monthly_report()` under-reports totals whenever a customer appears in more than one
region. The individual per-region rows look correct; only the aggregated totals are wrong.

## Misleading clue

`currency.py` rounds to 2 decimals on every conversion, and the discrepancy is small at
first glance. It looks like a floating-point or rounding problem. It is not: the totals are
wrong by exactly the duplicated amount, which rounding cannot explain.
