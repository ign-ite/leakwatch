"""Turn extracted terms into Tab 1's ids, and price terms into a dated price schedule."""
import re
from difflib import SequenceMatcher
from datetime import date, timedelta

import pandas as pd

from leakage.clean import match_vendor, normalize_vendor
from leakage.generate import ITEMS

ITEM_MATCH = 0.6  # below this, the item name is treated as unknown
_FILLER = {"per", "of", "pack", "a", "the"}
_SYNONYMS = {"kilogram": "kg"}


def _item_tokens(name: str) -> str:
    """'PET bottles, 500ml' and '500 ml PET bottles' -> '500 bottle ml pet'"""
    s = re.sub(r"(\d)([a-z])", r"\1 \2", name.lower())          # 500ml -> 500 ml
    words = re.sub(r"[^a-z0-9 ]", " ", s).split()
    words = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words]  # crude singular
    words = [_SYNONYMS.get(w, w) for w in words if w not in _FILLER]
    return " ".join(sorted(words))


_CATALOGUE = {_item_tokens(item): item for item in ITEMS}


def map_item(name: str) -> str | None:
    tokens = _item_tokens(name)
    score, item = max((SequenceMatcher(None, tokens, t).ratio(), i) for t, i in _CATALOGUE.items())
    return item if score >= ITEM_MATCH else None


def to_ids(terms: pd.DataFrame, vendors: pd.DataFrame) -> pd.DataFrame:
    """Replace free-text vendor and item names with vendor_id and catalogue item."""
    known = {normalize_vendor(n): v for n, v in zip(vendors["vendor_name"], vendors["vendor_id"])}
    out = terms.copy()
    out["vendor_id"] = [match_vendor(v, known)[0] for v in out["vendor"]]
    out["item"] = [map_item(i) for i in out["item"]]
    return out


def price_schedule(terms: pd.DataFrame) -> pd.DataFrame:
    """One row per (vendor, item, price period). A later effective price replaces the earlier
    one from its start date; an open-ended revision runs to the end of the contract."""
    prices = terms[terms["unit_price"].notna()].sort_values(["vendor_id", "item", "effective_from"])
    rows = []
    for (vendor_id, item), group in prices.groupby(["vendor_id", "item"], sort=False):
        contract_end = group["effective_to"].dropna().max()
        starts = list(group["effective_from"])
        for k, term in enumerate(group.itertuples(index=False)):
            if k + 1 < len(starts):
                end = (date.fromisoformat(starts[k + 1]) - timedelta(days=1)).isoformat()
            else:
                end = term.effective_to if pd.notna(term.effective_to) else contract_end
            rows.append({"vendor_id": vendor_id, "item": item, "unit_price": term.unit_price,
                         "valid_from": term.effective_from, "valid_to": end})
    return pd.DataFrame(rows, columns=["vendor_id", "item", "unit_price", "valid_from", "valid_to"])
