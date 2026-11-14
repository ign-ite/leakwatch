"""Turn three messy ERP exports into one clean invoice table."""
import re
from difflib import SequenceMatcher

import pandas as pd

AUTO_MATCH = 0.90  # at or above: same vendor, no human check needed
REVIEW_MATCH = 0.75  # between REVIEW_MATCH and AUTO_MATCH: matched, but flagged for review
_DROP_WORDS = {"pvt", "private", "ltd", "limited", "india"}
_OCR_FIXES = str.maketrans({"O": "0", "I": "1", "S": "5"})

# Each ERP names its columns differently; map them all to one schema.
ERP_COLUMNS = {
    "A": {"doc_id": "invoice_id", "supplier": "vendor_raw", "inv_no": "invoice_no",
          "pay_date": "paid_date", "material": "item", "quantity": "qty",
          "price": "unit_price", "total": "amount", "doc_type": "doc_type"},
    "B": {"ID": "invoice_id", "VENDOR_NAME": "vendor_raw", "INVOICE_NUMBER": "invoice_no",
          "PAYMENT_DATE": "paid_date", "ITEM": "item", "QTY": "qty",
          "UNIT_PRICE": "unit_price", "AMOUNT": "amount", "TYPE": "doc_type"},
    "C": {"record_id": "invoice_id", "vendor": "vendor_raw", "invoice_no": "invoice_no",
          "paid_on": "paid_date", "item": "item", "qty": "qty",
          "unit_price": "unit_price", "amount": "amount", "doc_type": "doc_type"},
}

UPLOAD_COLUMNS = ["invoice_id", "vendor_id", "invoice_no", "paid_date", "item",
                  "qty", "unit_price", "amount", "doc_type"]


def normalize_vendor(name: str) -> str:
    """'ACME (India) PVT. LTD.' -> 'acme'"""
    words = re.sub(r"[^a-z0-9 ]", " ", name.lower()).split()
    return " ".join(w for w in words if w not in _DROP_WORDS)


def match_vendor(name: str, known: dict[str, str]) -> tuple[str | None, float]:
    """Match a raw vendor name to a vendor_id.

    known maps normalized canonical name -> vendor_id.
    Returns (vendor_id or None, similarity score 0..1).
    """
    norm = normalize_vendor(name)
    if norm in known:
        return known[norm], 1.0
    best_id, best_score = None, 0.0
    for candidate, vendor_id in known.items():
        score = SequenceMatcher(None, norm, candidate).ratio()
        if score > best_score:
            best_id, best_score = vendor_id, score
    if best_score < REVIEW_MATCH:
        return None, best_score
    return best_id, best_score


def clean_invoice_no(raw: str) -> str:
    """Undo common OCR misreads so '0O45I23' and '0045123' compare equal."""
    s = re.sub(r"[^A-Z0-9]", "", raw.upper().translate(_OCR_FIXES))
    return s.lstrip("0") or "0"


def parse_eu_number(s: str) -> float:
    """'1.234,50' -> 1234.5"""
    return float(s.replace(".", "").replace(",", "."))


def clean_erp(raw: pd.DataFrame, erp: str) -> pd.DataFrame:
    df = raw.rename(columns=ERP_COLUMNS[erp])[list(ERP_COLUMNS[erp].values())].copy()
    if erp == "A":
        df["paid_date"] = pd.to_datetime(df["paid_date"], format="%d.%m.%Y")
        df["unit_price"] = df["unit_price"].map(parse_eu_number)
        df["amount"] = df["amount"].map(parse_eu_number)
    else:
        df["paid_date"] = pd.to_datetime(df["paid_date"], format="%Y-%m-%d")
        df["unit_price"] = df["unit_price"].astype(float)
        df["amount"] = df["amount"].astype(float)
    df["paid_date"] = df["paid_date"].dt.strftime("%Y-%m-%d")
    df["qty"] = df["qty"].astype(int)
    # Round money once here so the SQL can compare amounts with plain '='.
    df[["unit_price", "amount"]] = df[["unit_price", "amount"]].round(2)
    df["erp"] = erp
    return df


def clean_all(raw_erps: dict[str, pd.DataFrame], vendors: pd.DataFrame) -> pd.DataFrame:
    """Combine all ERPs into one table with vendor_id and invoice_no_clean."""
    df = pd.concat([clean_erp(raw, erp) for erp, raw in raw_erps.items()], ignore_index=True)
    known = {normalize_vendor(n): v for n, v in zip(vendors["vendor_name"], vendors["vendor_id"])}
    # ~200 distinct raw names vs 30k rows: match each distinct name once.
    matches = {name: match_vendor(name, known) for name in df["vendor_raw"].unique()}
    df["vendor_id"] = df["vendor_raw"].map(lambda n: matches[n][0])
    df["vendor_match_score"] = df["vendor_raw"].map(lambda n: round(matches[n][1], 3))
    df["vendor_needs_review"] = (
        df["vendor_id"].notna() & (df["vendor_match_score"] < AUTO_MATCH)
    ).astype(int)
    df["invoice_no_clean"] = df["invoice_no"].map(clean_invoice_no)
    return df


def prepare_upload(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a user CSV (read with dtype=str) and add the derived columns."""
    missing = [c for c in UPLOAD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    out = df[UPLOAD_COLUMNS].copy()
    out = out.apply(lambda col: col.str.strip()).replace("", None)
    blank = [c for c in UPLOAD_COLUMNS if out[c].isna().any()]
    if blank:
        raise ValueError("Blank values in: " + ", ".join(blank))
    out["doc_type"] = out["doc_type"].str.strip().str.upper()
    if not out["doc_type"].isin(["INV", "CRN"]).all():
        raise ValueError("doc_type must be INV or CRN")
    try:
        out["paid_date"] = pd.to_datetime(out["paid_date"], format="%Y-%m-%d").dt.strftime("%Y-%m-%d")
    except ValueError:
        raise ValueError("paid_date must be YYYY-MM-DD") from None
    try:
        out["qty"] = out["qty"].astype(int)
        out[["unit_price", "amount"]] = out[["unit_price", "amount"]].astype(float).round(2)
    except ValueError:
        raise ValueError("qty, unit_price and amount must be plain numbers like 1234.50") from None
    # Many ERPs store credit notes as positive amounts; the SQL expects them negative.
    is_credit = out["doc_type"] == "CRN"
    out.loc[is_credit, "amount"] = -out.loc[is_credit, "amount"].abs()
    out["invoice_no_clean"] = out["invoice_no"].map(clean_invoice_no)
    out["vendor_needs_review"] = 0
    return out
