"""Per-file parsing."""
import json
import logging

log = logging.getLogger(__name__)


def read_record(path: str) -> dict:
    """
    Read and parse a single JSON record file.

    Raises ValueError if the file is not a JSON object.
    """
    # BUG: the handle is opened without a context manager and is never closed on
    # the error path. Every malformed input leaks one file descriptor. Well-formed
    # inputs reach the end of the function and are closed by garbage collection
    # quickly enough that the leak is invisible until many files are processed.
    fh = open(path, "r", encoding="utf-8")
    raw = fh.read()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object in {path}")
    fh.close()
    return data


def normalise(record: dict) -> dict:
    """Normalise a record to the importer's canonical shape."""
    out = {}
    for key, value in record.items():
        k = key.strip().lower().replace(" ", "_")
        if isinstance(value, str):
            value = value.strip()
        out[k] = value
    return out
