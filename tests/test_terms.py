import pandas as pd
import pytest

from contracts.generate_contracts import ITEM_PROSE
from contracts.terms import map_item, price_schedule, to_ids

VENDORS = pd.DataFrame({"vendor_id": ["V001", "V002"],
                        "vendor_name": ["Konkan Agro Foods", "Bharat Tin Works"]})


@pytest.mark.parametrize("item, prose", [(i, p) for i, ps in ITEM_PROSE.items() for p in ps])
def test_map_item_recovers_every_prose_variant(item, prose):
    assert map_item(prose) == item


def test_map_item_unknown_is_none():
    assert map_item("diesel fuel") is None


def test_to_ids_maps_vendor_and_item_names():
    terms = pd.DataFrame({"vendor": ["Konkan Agro Foods Pvt. Ltd.", "Nobody Industries"],
                          "item": ["crown caps, packs of 1000", "malt (per kg)"]})
    out = to_ids(terms, VENDORS)
    assert out["vendor_id"][0] == "V001" and pd.isna(out["vendor_id"][1])
    assert list(out["item"]) == ["Crown caps (1000)", "Malt (kg)"]


def _price(unit_price, start, end, vendor_id="V001", item="Malt (kg)"):
    return {"vendor_id": vendor_id, "item": item, "unit_price": unit_price,
            "rebate_pct": None, "rebate_threshold_qty": None,
            "effective_from": start, "effective_to": end}


def test_price_schedule_latest_effective_price_wins():
    terms = pd.DataFrame([_price(100.0, "2025-04-01", "2026-03-31"),
                          _price(90.0, "2025-08-01", None)])
    out = price_schedule(terms)
    assert out[["unit_price", "valid_from", "valid_to"]].values.tolist() == [
        [100.0, "2025-04-01", "2025-07-31"], [90.0, "2025-08-01", "2026-03-31"]]


def test_price_schedule_open_ended_revision_inherits_contract_end():
    out = price_schedule(pd.DataFrame([_price(90.0, "2025-08-01", None),
                                       _price(100.0, "2025-04-01", "2026-03-31")]))
    assert out["valid_to"].tolist() == ["2025-07-31", "2026-03-31"]


def test_price_schedule_keeps_vendors_and_items_separate():
    out = price_schedule(pd.DataFrame([_price(100.0, "2025-04-01", "2026-03-31"),
                                       _price(5.0, "2025-04-01", "2026-03-31", item="Pallet wrap")]))
    assert len(out) == 2 and set(out["valid_to"]) == {"2026-03-31"}


def test_price_schedule_ignores_rebate_terms():
    rebate = _price(None, "2025-10-01", "2026-03-31") | {"rebate_pct": 3.0, "rebate_threshold_qty": 1000}
    assert len(price_schedule(pd.DataFrame([_price(100.0, "2025-04-01", "2026-03-31"), rebate]))) == 1
