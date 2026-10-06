"""Batch orchestration - contains a misleading threading red herring."""
import logging
from concurrent.futures import ThreadPoolExecutor

from reader import normalise, read_record
from sink import Sink

log = logging.getLogger(__name__)


def choose_workers(paths: list[str]) -> int:
    """
    Pick a worker count.

    This looks suspicious (it scales with the number of files) and it appears in the
    traceback, but it is NOT the cause of the descriptor leak - the leak happens in
    reader.read_record on the error path, independently of how many workers run.
    """
    return max(1, min(8, len(paths) // 4 or 1))


def import_paths(paths: list[str], sink: Sink) -> dict:
    """Import every path, returning a summary of successes and failures."""
    ok, failed = 0, []
    workers = choose_workers(paths)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(read_record, p): p for p in paths}
        for fut, path in futures.items():
            try:
                record = fut.result()
            except Exception as exc:  # noqa: BLE001
                log.warning("failed to read %s: %s", path, exc)
                failed.append({"path": path, "error": str(exc)})
                continue
            sink.write(normalise(record))
            ok += 1

    return {"ok": ok, "failed": failed, "workers": workers}
