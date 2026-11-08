"""1.10: a request may declare its currency, a rate table may say which it prices in, and the
result echoes the request's. Nothing converts, prices or ranks by it (docs/SERIALIZATION.md)."""

from __future__ import annotations

import copy
from typing import Any, Callable

import pytest

from packvium import (Container, Dimensions, InvalidRequestError, Item, Packer, PackingConfig,
                      pack_from_dict, validate)
from packvium.models import PackingRequest, RateTable

Json = dict[str, Any]

THREE_LETTERS = "must be three capital letters (ISO 4217)"


def json_request(currency: Any = None, table: Any = None) -> Json:
    request: Json = {
        "items": [{"id": "cube", "quantity": 2, "weight": "1000",
                   "dimensions": {"length": "100", "width": "100", "height": "100"}}],
        "containers": [{"id": "box", "inner_dimensions": {"length": "200", "width": "100", "height": "200"},
                        "rate_table": {"weight_brackets_g": [5000], "prices_minor": [900]}}],
        "configuration": {"solver_profile": "fast"},
    }
    if currency is not None:
        request["currency"] = currency
    if table is not None:
        request["containers"][0]["rate_table"]["currency"] = table
    return request


def json_refusal(request: Json) -> InvalidRequestError:
    with pytest.raises(InvalidRequestError) as caught:
        pack_from_dict(copy.deepcopy(request))
    return caught.value


def object_refusal(currency: Any = None, table: Any = None) -> InvalidRequestError:
    with pytest.raises(InvalidRequestError) as caught:
        objects(currency, table)()
    return caught.value


def objects(currency: Any = None, table: Any = None) -> Callable[[], Any]:
    rates = RateTable(weight_brackets_g=(5000,), prices_minor=(900,), currency=table)
    items = [Item.create("cube", Dimensions.mm(100, 100, 100), weight=1000, quantity=2)]
    containers = [Container.create("box", Dimensions.mm(200, 100, 200), rate_table=rates)]
    return lambda: Packer(PackingConfig.fast()).pack(items, containers, (), currency)


def test_the_result_echoes_the_request_currency_on_the_answer_and_every_alternative() -> None:
    request = json_request("EUR", "EUR")
    request["configuration"]["alternatives"] = 3
    result = pack_from_dict(request)
    assert result["currency"] == "EUR"
    assert all(alternative["currency"] == "EUR" for alternative in result["alternatives"])
    assert validate(request, result) == []


def test_a_request_without_a_currency_keeps_its_bytes() -> None:
    assert "currency" not in pack_from_dict(json_request())
    assert "currency" not in pack_from_dict({**json_request(), "currency": None})


def test_the_currency_changes_nothing_but_the_echo() -> None:
    plain, declared = pack_from_dict(json_request()), pack_from_dict(json_request("JPY"))
    for result in (plain, declared):
        result["algorithm"]["duration_ms"] = 0
    assert {key: value for key, value in declared.items() if key != "currency"} == plain


@pytest.mark.parametrize("currency, table, reason, field, detail", [
    (978, None, "wrong_type", "/currency", "must be a string"),
    ("usd", None, "invalid_value", "/currency", THREE_LETTERS),
    ("USD\n", None, "invalid_value", "/currency", THREE_LETTERS),
    ("ÜSD", None, "invalid_value", "/currency", THREE_LETTERS),
    ("USD", 840, "wrong_type", "/containers/0/rate_table/currency", "must be a string"),
    ("USD", "EURO", "invalid_value", "/containers/0/rate_table/currency", THREE_LETTERS),
    (None, "USD", "missing_field", "/currency", "is required when a rate_table declares a currency"),
    ("USD", "EUR", "not_allowed", "/containers/0/rate_table/currency",
     'must equal the request currency "USD"'),
])
def test_both_paths_refuse_a_bad_currency_alike(currency: Any, table: Any, reason: str, field: str,
                                                detail: str) -> None:
    for error in (json_refusal(json_request(currency, table)), object_refusal(currency, table)):
        assert (error.reason, error.field, error.detail) == (reason, field, detail)


def test_a_rate_table_may_leave_its_currency_to_the_request() -> None:
    assert objects("USD")().currency == "USD"
    assert objects("USD", "USD")().currency == "USD"
    assert objects()().currency is None


def test_the_object_api_carries_the_currency_last_and_optional() -> None:
    rates = RateTable((1000,), (500,))
    assert rates.currency is None
    request = PackingRequest((Item.create("cube", Dimensions.mm(1, 1, 1)),),
                             (Container.create("box", Dimensions.mm(2, 2, 2)),))
    assert request.currency is None
    assert PackingRequest(request.items, request.containers, (), "EUR").currency == "EUR"
