import pandas as pd

from leakage.score import add_priority, decoys_ignored, scorecard, top_n_share, value_caught


def findings(rows):
    return pd.DataFrame(rows, columns=["invoice_id", "check_type", "recoverable", "vendor_needs_review"])


ANSWER = pd.DataFrame([("B-1", "exact_duplicate", 100.0), ("B-2", "exact_duplicate", 300.0),
                       ("C-1", "ocr_duplicate", 50.0), ("D-1", "decoy", 0.0)],
                      columns=["invoice_id", "type", "amount"])


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


def test_decoys_ignored():
    assert decoys_ignored(findings([("B-1", "exact_duplicate", 1.0, 0)]), ANSWER) == (1, 1)
    assert decoys_ignored(findings([("D-1", "exact_duplicate", 1.0, 0)]), ANSWER) == (0, 1)


def test_value_caught():
    f = findings([("B-2", "exact_duplicate", 300.0, 0)])
    assert value_caught(f, ANSWER) == round(300 / 450, 3)


def test_top_n_share():
    ranked = add_priority(findings([("X", "exact_duplicate", 600.0, 0),
                                    ("Y", "exact_duplicate", 300.0, 0),
                                    ("Z", "exact_duplicate", 100.0, 0)]))
    assert top_n_share(ranked, 1) == 0.6
