"""The request error type, with no dependencies, so the models that refuse a value can raise it.

`request_errors` re-exports `InvalidRequestError` and holds the rules that raise it over a JSON
request; this module only defines the type and the form a model's own refusal takes.
"""

from __future__ import annotations


class InvalidRequestError(ValueError):
    """The request is not one any engine may answer. Nothing was solved."""

    code = "invalid_request"

    def __init__(self, reason: str, field: str, detail: str) -> None:
        self.reason = reason
        self.field = field
        self.detail = detail
        super().__init__(self._message())

    def _message(self) -> str:
        return f"{self.code}: {self.field}: {self.detail}" if self.field else f"{self.code}: {self.detail}"


class ObjectRefusal(InvalidRequestError):
    """A model refused a value while being built on its own, before it is part of a request, so
    there is no field to name; `Packer.pack` names the field for what it checks itself.

    The JSON path reports a refusal its rules did not name as `invalid_value` with this detail,
    the form every engine shares, so `serialization` turns one of these back into that."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(reason, "", detail)

    def _message(self) -> str:
        # The text a model has always raised, so a wrapper that quotes it (a commerce document,
        # a fixed placement's coordinate check) reads exactly what it read before.
        return self.detail


class ObjectTypeRefusal(ObjectRefusal, TypeError):
    """A mistyped value: still a `TypeError`, as before it was a request error."""


def refused(reason: str, detail: str) -> ObjectRefusal:
    return ObjectRefusal(reason, detail)


def refused_type(detail: str) -> ObjectTypeRefusal:
    return ObjectTypeRefusal("wrong_type", detail)
