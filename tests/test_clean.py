import pandas as pd
import pytest

from leakage.clean import (clean_all, clean_erp, clean_invoice_no, match_vendor,
                           normalize_vendor, parse_eu_number, prepare_upload)

KNOWN = {"konkan agro foods": "V001", "bharat tin works": "V002"}


@pytest.mark.parametrize("raw, expected", [
    ("Konkan Agro Foods", "konkan agro foods"),
    ("KONKAN AGRO FOODS PVT LTD", "konkan agro foods"),
    ("Konkan Agro Foods (India) Private Limited", "konkan agro foods"),
    ("Konkan Agro Foods Pvt. Ltd.", "konkan agro foods"),
])
def test_normalize_vendor(raw, expected):
    assert normalize_vendor(raw) == expected


def test_match_vendor_exact_after_normalizing():
    assert match_vendor("BHARAT TIN WORKS PVT LTD", KNOWN) == ("V002", 1.0)


def test_match_vendor_typo_auto_matches():
    vendor_id, score = match_vendor("Konkan Agro Fods Private Limited", KNOWN)
    assert vendor_id == "V001" and score >= 0.90


def test_match_vendor_abbreviation_needs_review():
    vendor_id, score = match_vendor("Konkan Agro F. Private Limited", KNOWN)
    assert vendor_id == "V001" and 0.75 <= score < 0.90


def test_match_vendor_unknown_is_unmatched():
    vendor_id, score = match_vendor("Totally Different Name", KNOWN)
    assert vendor_id is None and score < 0.75


@pytest.mark.parametrize("raw, expected", [
    ("0045123", "45123"),
    ("OO45123", "45123"),   # 0 read as O
    ("0045I23", "45123"),   # 1 read as I
    ("004S123", "45123"),   # 5 read as S
    ("045123", "45123"),    # leading zero dropped
    ("00-45 123", "45123"),  # punctuation and spaces stripped
])
def test_clean_invoice_no(raw, expected):
    assert clean_invoice_no(raw) == expected


def test_parse_eu_number():
    assert parse_eu_number("1.234,50") == 1234.5
    assert parse_eu_number("-42,00") == -42.0


def test_clean_erp_a_european_formats():
    raw = pd.DataFrame({"doc_id": ["A-1"], "supplier": ["Konkan Agro Foods"], "inv_no": ["0045123"],
                        "pay_date": ["14.03.2026"], "material": ["Malt (kg)"], "quantity": ["10"],
                        "price": ["1.234,50"], "total": ["12.345,00"], "doc_type": ["INV"]})
    row = clean_erp(raw, "A").iloc[0]
    assert row["paid_date"] == "2026-03-14"
    assert row["unit_price"] == 1234.5 and row["amount"] == 12345.0 and row["qty"] == 10
    assert row["invoice_no"] == "0045123" and row["erp"] == "A"


def test_clean_erp_b_iso_formats():
    raw = pd.DataFrame({"ID": ["B-1"], "VENDOR_NAME": ["BHARAT TIN WORKS PVT LTD"],
                        "INVOICE_NUMBER": ["0045123"], "PAYMENT_DATE": ["2026-03-14"],
                        "ITEM": ["Malt (kg)"], "QTY": ["10"], "UNIT_PRICE": ["1234.50"],
                        "AMOUNT": ["12345.00"], "TYPE": ["INV"]})
    row = clean_erp(raw, "B").iloc[0]
    assert row["paid_date"] == "2026-03-14" and row["unit_price"] == 1234.5
    assert row["vendor_raw"] == "BHARAT TIN WORKS PVT LTD"


def test_clean_all_adds_vendor_and_clean_invoice_no():
    vendors = pd.DataFrame({"vendor_id": ["V001", "V002"],
                            "vendor_name": ["Konkan Agro Foods", "Bharat Tin Works"]})
    c = pd.DataFrame({"record_id": ["C-1", "C-2"],
                      "vendor": ["Konkan Agro F. Private Limited", "Bharat Tin Works Private Limited"],
                      "invoice_no": ["OO45123", "0000777"], "paid_on": ["2026-01-01", "2026-01-02"],
                      "item": ["Malt (kg)", "Malt (kg)"], "qty": ["1", "2"],
                      "unit_price": ["5.00", "5.00"], "amount": ["5.00", "10.00"], "doc_type": ["INV", "INV"]})
    df = clean_all({"C": c}, vendors)
    assert list(df["vendor_id"]) == ["V001", "V002"]
    assert list(df["vendor_needs_review"]) == [1, 0]
    assert list(df["invoice_no_clean"]) == ["45123", "777"]


def test_prepare_upload_lists_missing_columns():
    with pytest.raises(ValueError, match="Missing columns: vendor_id, amount"):
        prepare_upload(pd.DataFrame(columns=["invoice_id", "invoice_no", "paid_date", "item",
                                             "qty", "unit_price", "doc_type"]))


def test_prepare_upload_adds_derived_columns():
    df = pd.DataFrame({"invoice_id": ["X1"], "vendor_id": ["V9"], "invoice_no": ["0O12"],
                       "paid_date": ["2026-02-01"], "item": ["Widget"], "qty": ["3"],
                       "unit_price": ["2.50"], "amount": ["7.50"], "doc_type": ["INV"]})
    row = prepare_upload(df).iloc[0]
    assert row["invoice_no_clean"] == "12" and row["vendor_needs_review"] == 0 and row["amount"] == 7.5


def test_clean_erp_rounds_money_to_two_decimals():
    raw = pd.DataFrame({"ID": ["B-1"], "VENDOR_NAME": ["X"], "INVOICE_NUMBER": ["1"],
                        "PAYMENT_DATE": ["2026-03-14"], "ITEM": ["Malt (kg)"], "QTY": ["3"],
                        "UNIT_PRICE": ["0.1001"], "AMOUNT": ["0.3003"], "TYPE": ["INV"]})
    row = clean_erp(raw, "B").iloc[0]
    assert row["unit_price"] == 0.1 and row["amount"] == 0.3


def _upload(**overrides):
    row = {"invoice_id": "X1", "vendor_id": "V9", "invoice_no": "0O12", "paid_date": "2026-02-01",
           "item": "Widget", "qty": "3", "unit_price": "2.50", "amount": "7.50", "doc_type": "INV"}
    row.update(overrides)
    return pd.DataFrame([row])


@pytest.mark.parametrize("column", ["invoice_no", "vendor_id", "qty"])
def test_prepare_upload_rejects_blank_required_values(column):
    with pytest.raises(ValueError, match=f"Blank values in: {column}"):
        prepare_upload(_upload(**{column: None}))


def test_prepare_upload_accepts_lowercase_doc_type():
    assert prepare_upload(_upload(doc_type=" inv ")).iloc[0]["doc_type"] == "INV"


def test_prepare_upload_rejects_unknown_doc_type():
    with pytest.raises(ValueError, match="doc_type must be INV or CRN"):
        prepare_upload(_upload(doc_type="PAYMENT"))


def test_prepare_upload_rejects_non_iso_dates():
    with pytest.raises(ValueError, match="paid_date must be YYYY-MM-DD"):
        prepare_upload(_upload(paid_date="05/01/2026"))


def test_prepare_upload_rejects_formatted_numbers():
    with pytest.raises(ValueError, match="plain numbers"):
        prepare_upload(_upload(amount="1,234.50"))


def test_prepare_upload_treats_whitespace_as_blank():
    with pytest.raises(ValueError, match="Blank values in: invoice_no"):
        prepare_upload(_upload(invoice_no="   "))


def test_prepare_upload_makes_credit_note_amounts_negative():
    # many ERPs store credit notes as positive amounts; the checks expect negatives
    out = prepare_upload(_upload(doc_type="CRN", amount="7.50"))
    assert out.iloc[0]["amount"] == -7.5
