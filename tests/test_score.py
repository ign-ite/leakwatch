import pandas as pd

from leakage.score import (add_priority, miss_reasons, scorecard, top_n_share, traps_ignored,
                           value_caught)


def findings(rows):
    return pd.DataFrame(rows, columns=["invoice_id", "check_type", "recoverable", "vendor_needs_review"])


ANSWER = pd.DataFrame([("B-1", "exact_duplicate", 100.0, "A-1"), ("B-2", "exact_duplicate", 300.0, "A-2"),
                       ("C-1", "ocr_duplicate", 50.0, "A-3"), ("D-1", "decoy", 0.0, "A-4"),
                       ("E-1", "reused_number", 0.0, "A-5")],
                      columns=["invoice_id", "type", "amount", "original_id"])


def test_add_priority_confidence_and_order():
    out = add_priority(findings([("X", "ocr_duplicate", 100.0, 0),
                                 ("Y", "exact_duplicate", 100.0, 1),
                                 ("Z", "price_overcharge", 500.0, 0)]))
    assert list(out["invoice_id"]) == ["Z", "X", "Y"]
    assert list(out["confidence"]) == [1.0, 0.8, 0.8]
    assert list(out["priority"]) == [500.0, 80.0, 80.0]


def test_scorecard_precision_recall():
    f = findings([("B-1", "exact_duplicate", 100.0, 0), ("Q-9", "exact_duplicate", 10.0, 0)])
    card = scorecard(f, ANSWER).set_index("check")
    assert card.loc["exact_duplicate", "precision"] == 0.5   # 1 of 2 flags real
    assert card.loc["exact_duplicate", "recall"] == 0.5      # 1 of 2 planted found
    assert card.loc["ocr_duplicate", "recall"] == 0.0
    assert pd.isna(card.loc["ocr_duplicate", "precision"])   # nothing flagged


def test_traps_ignored_per_type():
    out = traps_ignored(findings([("E-1", "exact_duplicate", 1.0, 0)]), ANSWER).set_index("trap")
    assert out.loc["decoy", "ignored"] == 1 and out.loc["decoy", "total"] == 1
    assert out.loc["reused_number", "ignored"] == 0


def test_value_caught_needs_the_right_check_type():
    f = findings([("B-2", "exact_duplicate", 300.0, 0), ("C-1", "exact_duplicate", 50.0, 0)])
    assert value_caught(f, ANSWER) == round(300 / 450, 3)   # C-1 is OCR, flagged as exact


def test_miss_reasons():
    invoices = pd.DataFrame({
        "invoice_id": ["A-1", "B-1", "A-2", "B-2", "A-3", "C-1"],
        "vendor_id":  ["V1",  None,  "V1",  "V1",  "V1",  "V1"],
        "paid_date":  ["2026-01-01", "2026-01-05", "2026-01-01", "2026-01-05",
                       "2026-01-01", "2026-03-01"],
    })
    out = miss_reasons(findings([("B-2", "exact_duplicate", 300.0, 0)]), ANSWER, invoices)
    assert sorted(map(tuple, out[["check", "reason", "missed"]].values.tolist())) == [
        ("exact_duplicate", "vendor name unmatched", 1),
        ("ocr_duplicate", "paid more than 30 days after the original", 1)]


def test_top_n_share():
    ranked = add_priority(findings([("X", "exact_duplicate", 600.0, 0),
                                    ("Y", "exact_duplicate", 300.0, 0),
                                    ("Z", "exact_duplicate", 100.0, 0)]))
    assert top_n_share(ranked, 1) == 0.6
