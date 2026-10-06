"""Report aggregation."""

from currency import to_cents


def monthly_report(rows: list[dict]) -> dict:
    """
    Aggregate per-region sales rows into report totals.

    Each row: {"region": str, "customer_id": str, "amount": float}

    Returns {"by_region": {...}, "by_customer": {...}, "grand_total": float}
    """
    by_region: dict[str, int] = {}
    by_customer: dict[str, int] = {}

    for row in rows:
        region = row["region"]
        customer = row["customer_id"]
        cents = to_cents(row["amount"])

        by_region[region] = by_region.get(region, 0) + cents

        # BUG: a customer that appears in several regions must be summed, not
        # overwritten. Assignment keeps only the last region's amount, so the
        # grand total silently under-reports every multi-region customer.
        by_customer[customer] = cents

    grand_total = sum(by_customer.values())

    return {
        "by_region": {k: v / 100 for k, v in by_region.items()},
        "by_customer": {k: v / 100 for k, v in by_customer.items()},
        "grand_total": grand_total / 100,
    }
