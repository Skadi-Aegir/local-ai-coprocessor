# Project 2: batch file importer

**Fixed fixture for Phase 3 Test A.** Contains one real bug and one misleading clue.

## Symptom

The importer works for the first few files, then fails with
`OSError: [Errno 24] Too many open files`. Restarting fixes it temporarily.
The leak only appears when some inputs are malformed.

## Misleading clue

`batch.py` creates a `ThreadPoolExecutor` with `max_workers` derived from the file count, and
the traceback surfaces while threads are being submitted. It looks like a threading problem.
It is not.
