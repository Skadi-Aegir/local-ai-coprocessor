# Project 1: pagination service

This is a **fixed fixture for Phase 3 Test A**. It contains exactly one real bug and one
misleading clue. Do not "fix" it — the benchmark applies model patches to a pristine copy.

## Symptom

`GET /items?page=1` returns the *second* page of results. `page=0` returns the first page.
The unit tests pass because they assert on `total_pages` rather than on returned items.

## Misleading clue

`storage.py` logs `WARNING: page beyond range` even for valid requests, and the log line
appears immediately before the wrong results. It looks like the storage layer is at fault.
