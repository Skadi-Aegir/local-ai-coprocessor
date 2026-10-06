"""Input loading for the report."""
import csv


def load_rows(path: str) -> list[dict]:
    """Load report rows from a CSV with header region,customer_id,amount."""
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append({
                "region": row["region"],
                "customer_id": row["customer_id"],
                "amount": float(row["amount"]),
            })
    return out
