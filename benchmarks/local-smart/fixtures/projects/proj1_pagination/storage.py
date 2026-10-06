"""In-memory item storage with a deliberately noisy, misleading log line."""
import logging

log = logging.getLogger(__name__)

_ITEMS = [{"id": i, "name": f"item-{i}"} for i in range(1, 26)]


class Store:
    def __init__(self, items=None):
        self._items = list(items if items is not None else _ITEMS)

    def __len__(self) -> int:
        return len(self._items)

    def fetch(self, start: int, end: int) -> list[dict]:
        """
        Return the items in the half-open range [start, end).

        Emits a warning when the requested range looks out of the ordinary.
        This warning fires for perfectly valid requests and is NOT the cause of
        the pagination symptom - it is a red herring placed here on purpose.
        """
        if start >= len(self._items):
            log.warning("page beyond range: start=%s len=%s", start, len(self._items))
        elif start > 0:
            # fires on every request that is not the very first page
            log.warning("page beyond range: start=%s len=%s", start, len(self._items))

        return self._items[start:end]
