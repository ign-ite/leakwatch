from leakage.load import run_check
from tests.conftest import invoice


def flagged(conn, check):
    return sorted(run_check(conn, check)["invoice_id"])


def test_exact_duplicates_flags_later_copy_only(make_db):
    conn = make_db([
        invoice("A-1", "0045123", "2026-01-05"),
        invoice("B-1", "0045123", "2026-02-01"),                   # repeat payment -> flag
        invoice("A-2", "0045124", "2026-01-06"),                   # different number
        invoice("A-3", "0045125", "2026-01-07", amount=100.0),
        invoice("B-2", "0045125", "2026-01-20", amount=250.0),     # same number, other amount
        invoice("C-1", "0045123", "2026-01-05", vendor_id="V002"), # other vendor
    ])
    assert flagged(conn, "exact_duplicates") == ["B-1"]


def test_exact_duplicates_skips_pairs_reversed_by_credit_note(make_db):
    conn = make_db([
        invoice("A-1", "0045123", "2026-01-05"),
        invoice("B-1", "0045123", "2026-02-01"),
        invoice("B-2", "0045123", "2026-02-10", amount=-100.0, doc_type="CRN"),
    ])
    assert flagged(conn, "exact_duplicates") == []


def test_exact_duplicates_reason_names_first_payment(make_db):
    conn = make_db([invoice("A-1", "0045123", "2026-01-05"), invoice("B-1", "0045123", "2026-02-01")])
    reason = run_check(conn, "exact_duplicates").loc[0, "reason"]
    assert "A-1" in reason and "2026-01-05" in reason


def test_ocr_duplicates_flags_misread_within_30_days(make_db):
    conn = make_db([
        invoice("A-1", "0045123", "2026-01-05", clean="45123"),
        invoice("C-1", "OO45123", "2026-01-20", clean="45123"),   # misread, 15 days -> flag
        invoice("A-2", "0099001", "2026-01-05", clean="99001"),
        invoice("C-2", "0O99001", "2026-03-01", clean="99001"),   # misread but 55 days -> skip
        invoice("A-3", "0077001", "2026-01-05", clean="77001"),
        invoice("B-3", "0077001", "2026-01-10", clean="77001"),   # identical number: exact dup, not OCR
    ])
    assert flagged(conn, "ocr_duplicates") == ["C-1"]


def test_ocr_duplicates_skips_credit_noted(make_db):
    conn = make_db([
        invoice("A-1", "0045123", "2026-01-05", clean="45123"),
        invoice("C-1", "OO45123", "2026-01-20", clean="45123"),
        invoice("C-2", "0045123", "2026-01-25", clean="45123", amount=-100.0, doc_type="CRN"),
    ])
    assert flagged(conn, "ocr_duplicates") == []


def test_price_overcharge_flags_and_computes_recoverable(make_db):
    conn = make_db([
        invoice("A-1", "1", "2026-01-05", unit_price=10.0, qty=10, amount=100.0),   # at contract
        invoice("A-2", "2", "2026-01-05", unit_price=12.5, qty=10, amount=125.0),   # overcharged
        invoice("A-3", "3", "2026-01-05", unit_price=9.0, qty=10, amount=90.0),     # under contract
        invoice("A-4", "4", "2026-01-05", unit_price=99.0, item="No contract item"),
        invoice("A-5", "5", "2026-01-05", unit_price=12.5, doc_type="CRN", amount=-125.0),
    ])
    result = run_check(conn, "price_overcharge")
    assert list(result["invoice_id"]) == ["A-2"]
    assert result.loc[0, "recoverable"] == 25.0
