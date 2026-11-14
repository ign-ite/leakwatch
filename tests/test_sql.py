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
