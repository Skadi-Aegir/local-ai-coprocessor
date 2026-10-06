"""FIXTURE for Phase 3 Test E. Existing tests must all keep passing."""


def test_add_all_empty():
    from service import add_all
    assert add_all([]) == 0


def test_add_all_values():
    from service import add_all
    assert add_all([1, 2, 3]) == 6
    assert add_all([-1, 1]) == 0


def test_public_entrypoint_shape():
    from service import public_entrypoint
    out = public_entrypoint([4, 5])
    assert set(out.keys()) == {"sum", "last"}
    assert out["sum"] == 9
