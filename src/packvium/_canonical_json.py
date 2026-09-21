"""RFC 8785 canonical JSON: the one spelling four engines can agree on byte for byte.

A document that is compared, signed or replayed across Python, PHP, Rust and JavaScript needs
one serialization, and every language's default JSON writer disagrees with the others
somewhere. Python prints `1.0` where JavaScript prints `1`; JavaScript sorts keys by UTF-16
code unit and Python by code point; PHP escapes U+2028 unless told not to. RFC 8785 settles
each of those, and it settles them the way JavaScript already behaves -- which matters,
because JavaScript is the one engine that cannot tell `1.0` from `1` once the text is parsed.

- Object keys are sorted by their UTF-16 code units.
- Strings escape `"`, `\\`, and U+0000..U+001F (`\\b \\t \\n \\f \\r` by name, the rest as
  lowercase `\\u00xx`); everything else is written as UTF-8.
- Numbers are written as ECMAScript's `Number::toString` writes them.
- No whitespace.

Two refusals keep that promise honest. A number whose magnitude exceeds 2^53 - 1 (or is not
finite) is refused, because JavaScript already holds a different number by the time it can see
it. A string carrying a lone UTF-16 surrogate is refused, because it has no UTF-8 spelling.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any, List, Mapping

#: The largest magnitude every engine holds exactly.
MAX_EXACT_MAGNITUDE = 2**53 - 1

_NAMED_ESCAPES = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r"}


class CanonicalJsonError(ValueError):
    """A value has no canonical spelling. `code` names which rule it broke."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def canonical_json(value: Any) -> str:
    parts: List[str] = []
    _write(value, parts)
    return "".join(parts)


def _write(value: Any, parts: List[str]) -> None:
    if value is None:
        parts.append("null")
    elif value is True:
        parts.append("true")
    elif value is False:
        parts.append("false")
    elif isinstance(value, str):
        parts.append(_string(value))
    elif isinstance(value, int):
        parts.append(_integer(value))
    elif isinstance(value, float):
        parts.append(_number(value))
    elif isinstance(value, Mapping):
        _object(value, parts)
    elif isinstance(value, (list, tuple)):
        parts.append("[")
        for index, element in enumerate(value):
            if index:
                parts.append(",")
            _write(element, parts)
        parts.append("]")
    else:
        raise CanonicalJsonError("invalid_value", f"a {type(value).__name__} has no JSON spelling")


def _object(value: Mapping[Any, Any], parts: List[str]) -> None:
    spelled = {}
    for key in value:
        if not isinstance(key, str):
            raise CanonicalJsonError("invalid_value", f"object key {key!r} is not a string")
        spelled[key] = _string(key)
    parts.append("{")
    # Big-endian UTF-16 bytes compare exactly as UTF-16 code units do; `_string` has already
    # refused the lone surrogates that would not encode.
    for index, key in enumerate(sorted(spelled, key=lambda name: name.encode("utf-16-be"))):
        if index:
            parts.append(",")
        parts.append(spelled[key])
        parts.append(":")
        _write(value[key], parts)
    parts.append("}")


def _string(text: str) -> str:
    escaped = []
    for character in text:
        code = ord(character)
        if 0xD800 <= code <= 0xDFFF:
            raise CanonicalJsonError("invalid_string", "a string carries a lone UTF-16 surrogate")
        if character in _NAMED_ESCAPES:
            escaped.append(_NAMED_ESCAPES[character])
        elif code < 0x20:
            escaped.append(f"\\u{code:04x}")
        else:
            escaped.append(character)
    return '"' + "".join(escaped) + '"'


def _integer(value: int) -> str:
    if abs(value) > MAX_EXACT_MAGNITUDE:
        raise CanonicalJsonError("number_out_of_range", f"{value} is beyond what every engine holds exactly")
    return str(value)


def _number(value: float) -> str:
    if not math.isfinite(value) or abs(value) > MAX_EXACT_MAGNITUDE:
        raise CanonicalJsonError("number_out_of_range", f"{value!r} is beyond what every engine holds exactly")
    if value == 0:
        return "0"
    # `repr` is the shortest string that round-trips, which is the digit string ECMAScript uses.
    _, digit_tuple, exponent = Decimal(repr(abs(value))).as_tuple()
    digits = list(digit_tuple)
    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()
        exponent += 1
    text = "".join(str(digit) for digit in digits)
    count, point = len(digits), len(digits) + int(exponent)
    if count <= point <= 21:
        rendered = text + "0" * (point - count)
    elif 0 < point <= 21:
        rendered = text[:point] + "." + text[point:]
    elif -6 < point <= 0:
        rendered = "0." + "0" * -point + text
    else:
        power = point - 1
        rendered = text[0] + ("." + text[1:] if count > 1 else "") + "e" + ("+" if power >= 0 else "-") + str(abs(power))
    return ("-" if value < 0 else "") + rendered
