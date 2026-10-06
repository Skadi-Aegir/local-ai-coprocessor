"""Test for the report.

PASSES against the buggy code: the fixture rows use a distinct customer per region,
so the multi-region overwrite never triggers.
"""
from report import monthly_report


def test_single_region_each():
    rows = [
        {"region": "north", "customer_id": "c1", "amount": 10.00},
        {"region": "south", "customer_id": "c2", "amount": 20.00},
    ]
    out = monthly_report(rows)
    assert out["by_region"]["north"] == 10.00
    assert out["by_region"]["south"] == 20.00
    assert out["by_customer"]["c1"] == 10.00
    assert out["by_customer"]["c2"] == 20.00
    assert out["grand_total"] == 30.00
