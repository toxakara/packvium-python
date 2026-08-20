"""Error types for the exported commercial and control-plane API.

Two kinds of failure, deliberately kept apart (docs/COMMERCE-API.md, "Input errors
versus rejections"):

  * a **caller bug** -- a missing key, a negative weight, an unknown policy operator --
    is a `CommerceInputError`, raised the way Python reports caller bugs;
  * a **rejection** the commercial model is entitled to make -- no tariff effective at
    that instant, no rate for that zone -- is not an exception at all. It is a
    successful call returning a result document whose `status` is `"rejected"`, the
    same way an infeasible packing request returns a `PackingResult` with a status.

`_Rejection` is the internal carrier for the second kind between the point of detection
and the API boundary; it never escapes `packvium.commerce`.
"""

from __future__ import annotations

from typing import Any, Mapping


class CommerceError(Exception):
    """Base class for every error raised by `packvium.commerce`."""


class CommerceInputError(CommerceError):
    """The supplied document or request is not well formed."""


class _Rejection(Exception):
    """Internal: a structured rejection travelling to the API boundary, where it is
    turned into the `{"status": "rejected", ...}` result document."""

    def __init__(self, code: str, fields: Mapping[str, Any]) -> None:
        super().__init__(code)
        self.code = code
        self.fields = dict(fields)
