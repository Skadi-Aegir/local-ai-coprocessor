"""FIXTURE for Phase 3 Test E. Do not edit - the task explicitly protects this file.

Stable public interface. Changing either name or signature breaks downstream callers.
"""


def public_entrypoint(xs: list[int]) -> dict:
    """Stable API. MUST NOT change name, signature, or return keys."""
    raise NotImplementedError("re-exported from service.py")
