import pandas as pd
import pytest

from leakage.load import load_db, run_all, run_check

COLUMNS = ["invoice_id", "vendor_id", "vendor_needs_review", "invoice_no", "invoice_no_clean",
           "paid_date", "item", "qty", "unit_price", "amount", "doc_type"]


def invoice(invoice_id, invoice_no, paid_date, amount=100.0, vendor_id="V001", doc_type="INV",
            clean=None, item="Malt (kg)", qty=10, unit_price=10.0, review=0):
    return [invoice_id, vendor_id, review, invoice_no, clean or invoice_no.lstrip("0"),
            paid_date, item, qty, unit_price, amount, doc_type]


@pytest.fixture
def make_db():
    """make_db(rows, contracts=None) -> sqlite connection with invoices + contracts tables."""
    def _make(rows, contracts=None):
        invoices = pd.DataFrame(rows, columns=COLUMNS)
        contracts = contracts if contracts is not None else pd.DataFrame(
            {"vendor_id": ["V001"], "item": ["Malt (kg)"], "contract_price": [10.0]})
        return load_db(invoices, contracts)
    return _make


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


def test_ocr_window_boundary_is_inclusive_at_30_days(make_db):
    conn = make_db([
        invoice("A-1", "0045123", "2026-01-01", clean="45123"),
        invoice("C-1", "OO45123", "2026-01-31", clean="45123"),   # exactly 30 days -> flag
        invoice("A-2", "0099001", "2026-01-01", clean="99001"),
        invoice("C-2", "0O99001", "2026-02-01", clean="99001"),   # 31 days -> skip
    ])
    assert flagged(conn, "ocr_duplicates") == ["C-1"]


def test_unmatched_vendor_rows_are_not_checked(make_db):
    conn = make_db([
        invoice("A-1", "0045123", "2026-01-05", vendor_id=None),
        invoice("B-1", "0045123", "2026-02-01", vendor_id=None),
    ])
    assert run_all(conn).empty


def test_invoice_paid_three_ways_is_flagged_once_per_extra_payment(make_db):
    # original, then an OCR misread copy, then an exact copy: 2 overpayments, 2 findings
    conn = make_db([
        invoice("A-1", "0045123", "2026-01-01", clean="45123"),
        invoice("C-1", "OO45123", "2026-01-05", clean="45123"),
        invoice("B-1", "0045123", "2026-01-10", clean="45123"),
    ])
    found = run_all(conn)
    assert sorted(zip(found["invoice_id"], found["check_type"])) == [
        ("B-1", "exact_duplicate"), ("C-1", "ocr_duplicate")]


def test_one_credit_note_cancels_only_one_repeat_payment(make_db):
    # paid three times, one credit note: one overpayment is still outstanding
    conn = make_db([
        invoice("A-1", "0045123", "2026-01-01"),
        invoice("B-1", "0045123", "2026-01-10"),
        invoice("C-1", "0045123", "2026-01-20"),
        invoice("C-2", "0045123", "2026-01-25", amount=-100.0, doc_type="CRN"),
    ])
    assert flagged(conn, "exact_duplicates") == ["C-1"]


def test_credit_note_on_mixed_duplicates_leaves_one_flag(make_db):
    conn = make_db([
        invoice("A-1", "0045123", "2026-01-01", clean="45123"),
        invoice("C-1", "OO45123", "2026-01-05", clean="45123"),
        invoice("B-1", "0045123", "2026-01-10", clean="45123"),
        invoice("B-2", "0045123", "2026-01-12", clean="45123", amount=-100.0, doc_type="CRN"),
    ])
    assert len(run_all(conn)) == 1
