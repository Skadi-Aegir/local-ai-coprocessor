"""Tests for the pagination service.

These tests PASS against the buggy code, because they only assert on metadata
(total_pages, page echo) and never on which items are actually returned.
"""
from pagination import total_pages
from service import ItemService
from storage import Store


def test_total_pages():
    assert total_pages(25, 10) == 3
    assert total_pages(20, 10) == 2
    assert total_pages(0, 10) == 0


def test_metadata_echo():
    svc = ItemService(Store())
    out = svc.list_items({"page": "2", "per_page": "10"})
    assert out["page"] == 2
    assert out["per_page"] == 10
    assert out["total"] == 25
    assert out["total_pages"] == 3


def test_validation_rejects_bad_page():
    import pytest
    from validators import ValidationError
    svc = ItemService(Store())
    with pytest.raises(ValidationError):
        svc.list_items({"page": "0"})
