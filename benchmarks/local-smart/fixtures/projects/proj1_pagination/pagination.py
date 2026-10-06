"""Pagination helpers."""


def page_bounds(page: int, per_page: int, total: int) -> tuple[int, int]:
    """
    Return the half-open slice [start, end) of items for the given page.

    Pages are 1-based as presented to the API caller.
    """
    # BUG: `page` is 1-based, so the first page must start at 0.
    # Subtracting nothing means page=1 skips the first per_page items.
    start = page * per_page
    end = min(start + per_page, total)
    return start, end


def total_pages(total: int, per_page: int) -> int:
    """Number of pages needed to hold `total` items."""
    if per_page <= 0:
        raise ValueError("per_page must be positive")
    return (total + per_page - 1) // per_page
