"""Query validation."""
MAX_PER_PAGE = 100


class ValidationError(ValueError):
    pass


def validate_query(query: dict) -> None:
    """Validate an incoming query dict. Raises ValidationError on bad input."""
    if "page" in query:
        try:
            page = int(query["page"])
        except (TypeError, ValueError):
            raise ValidationError("page must be an integer") from None
        if page < 1:
            raise ValidationError("page must be >= 1")

    if "per_page" in query:
        try:
            per_page = int(query["per_page"])
        except (TypeError, ValueError):
            raise ValidationError("per_page must be an integer") from None
        if per_page < 1:
            raise ValidationError("per_page must be >= 1")
        if per_page > MAX_PER_PAGE:
            raise ValidationError(f"per_page must be <= {MAX_PER_PAGE}")
