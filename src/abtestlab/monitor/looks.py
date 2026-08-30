"""Turn a raw experiment observation log into an ordered sequence of looks.

A *look* is the cumulative state of both arms at one point in time -- the thing
an analyst actually sees when they open the dashboard on day 3. Everything
downstream (SRM gate, sequential test, stopping policy) is defined over this
sequence, so the parsing rules here are the first line of defence against a
broken export.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

REQUIRED_FIELDS = (
    "conversions_control",
    "n_control",
    "conversions_treatment",
    "n_treatment",
)


class LookSequenceError(ValueError):
    """Raised when an observation log cannot be read as an ordered experiment."""


@dataclass(frozen=True)
class Look:
    """Cumulative counts for both arms at one monitoring checkpoint.

    Attributes:
        index: Zero-based position in the log.
        label: Human label for the checkpoint (a date, a day number, a batch id).
        conversions_control: Cumulative control conversions.
        n_control: Cumulative control exposures.
        conversions_treatment: Cumulative treatment conversions.
        n_treatment: Cumulative treatment exposures.
    """

    index: int
    label: str
    conversions_control: int
    n_control: int
    conversions_treatment: int
    n_treatment: int

    @property
    def total_units(self) -> int:
        """Units exposed across both arms up to and including this look."""
        return self.n_control + self.n_treatment

    @property
    def rate_control(self) -> float:
        """Cumulative control conversion rate."""
        return self.conversions_control / self.n_control

    @property
    def rate_treatment(self) -> float:
        """Cumulative treatment conversion rate."""
        return self.conversions_treatment / self.n_treatment


def _as_int(row: Mapping[str, Any], field: str, label: str) -> int:
    """Read one integer count out of a log row.

    Args:
        row: Raw log row.
        field: Field name to read.
        label: Checkpoint label, used in error messages.

    Returns:
        The parsed count.

    Raises:
        LookSequenceError: If the field is missing, non-numeric, or negative.
    """
    if field not in row:
        raise LookSequenceError(f"look {label!r} is missing required field {field!r}")
    try:
        value = int(row[field])
    except (TypeError, ValueError) as exc:
        raise LookSequenceError(f"look {label!r} field {field!r} is not an integer") from exc
    if value < 0:
        raise LookSequenceError(f"look {label!r} field {field!r} is negative ({value})")
    return value


def build_looks(rows: Sequence[Mapping[str, Any]], *, cumulative: bool = True) -> tuple[Look, ...]:
    """Read an observation log into a validated, ordered sequence of looks.

    Args:
        rows: Log rows in chronological order. Each row needs the four count
            fields; ``label`` is optional and defaults to ``look-<i>``.
        cumulative: ``True`` if each row already holds running totals, ``False``
            if rows hold per-period increments that must be accumulated here.

    Returns:
        Cumulative looks in chronological order.

    Raises:
        LookSequenceError: If the log is empty, a count is malformed, a
            cumulative count moves backwards, an arm has no exposures at its
            first look, or conversions exceed exposures.
    """
    if not rows:
        raise LookSequenceError("observation log is empty; nothing to monitor")

    looks: list[Look] = []
    running = dict.fromkeys(REQUIRED_FIELDS, 0)
    previous = dict(running)

    for index, row in enumerate(rows):
        label = str(row.get("label", f"look-{index}"))
        values = {field: _as_int(row, field, label) for field in REQUIRED_FIELDS}
        if cumulative:
            running = values
            for field in REQUIRED_FIELDS:
                if running[field] < previous[field]:
                    raise LookSequenceError(
                        f"cumulative {field} went backwards at look {label!r}: "
                        f"{previous[field]} -> {running[field]}"
                    )
        else:
            running = {field: previous[field] + values[field] for field in REQUIRED_FIELDS}
        previous = running

        look = Look(index=index, label=label, **running)
        if look.n_control <= 0 or look.n_treatment <= 0:
            raise LookSequenceError(
                f"look {label!r} has an unexposed arm "
                f"(n_control={look.n_control}, n_treatment={look.n_treatment})"
            )
        if look.conversions_control > look.n_control:
            raise LookSequenceError(
                f"look {label!r}: {look.conversions_control} control conversions "
                f"from {look.n_control} exposures"
            )
        if look.conversions_treatment > look.n_treatment:
            raise LookSequenceError(
                f"look {label!r}: {look.conversions_treatment} treatment conversions "
                f"from {look.n_treatment} exposures"
            )
        looks.append(look)

    return tuple(looks)
