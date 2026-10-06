"""HTTP layer over the pagination service."""
from pagination import page_bounds, total_pages
from storage import Store
from validators import validate_query

DEFAULT_PER_PAGE = 10


class ItemService:
    def __init__(self, store: Store):
        self.store = store

    def list_items(self, query: dict) -> dict:
        validate_query(query)
        page = int(query.get("page", 1))
        per_page = int(query.get("per_page", DEFAULT_PER_PAGE))
        total = len(self.store)

        start, end = page_bounds(page, per_page, total)
        items = self.store.fetch(start, end)

        return {
            "items": items,
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages(total, per_page),
        }
