"""Test for the importer.

PASSES against the buggy code: it imports only well-formed inputs, which do not
trigger the descriptor leak. The bug needs malformed inputs to reproduce.
"""
import json
import os
import tempfile

from batch import import_paths
from sink import Sink


def _write(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(obj if isinstance(obj, str) else json.dumps(obj))


def test_imports_valid_records():
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i in range(3):
            p = os.path.join(d, f"r{i}.json")
            _write(p, {" Name ": f" item {i} ", "Count": i})
            paths.append(p)

        out = os.path.join(d, "out.jsonl")
        with Sink(out) as s:
            summary = import_paths(paths, s)

        assert summary["ok"] == 3
        assert summary["failed"] == []

        with open(out, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        assert rows[0]["name"] == "item 0"
        assert rows[0]["count"] == 0
