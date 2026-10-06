"""Currency conversion - the misleading clue lives here."""
from decimal import ROUND_HALF_UP, Decimal

# Fixed rate for this fixture; a real system would look this up.
RATE = Decimal("1.0")


def to_cents(amount: float) -> int:
    """
    Convert an amount to integer cents.

    The quantize() call makes this look like the source of small discrepancies.
    It is not: the report bug loses whole rows, not fractions of a cent, and the
    rate here is 1.0 so no conversion error is possible in this fixture.
    """
    value = (Decimal(str(amount)) * RATE).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return int(value * 100)
