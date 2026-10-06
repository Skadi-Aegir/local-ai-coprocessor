"""Output sink."""
import json
import os


class Sink:
    """Append normalised records to a JSON Lines file."""

    def __init__(self, path: str):
        self.path = path
        self._fh = None

    def write(self, record: dict) -> None:
        if self._fh is None:
            self._fh = open(self.path, "a", encoding="utf-8")
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
