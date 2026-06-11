"""Hermes HUD widgets."""

# Shared capacity bar color thresholds used across overview, profiles, and boot screen
CAPACITY_RED_PCT = 90
CAPACITY_YELLOW_PCT = 70


def escape_markup(text) -> str:
    """Escape [ in user data so Textual never interprets it as markup."""
    return str(text).replace("[", "\\[")
