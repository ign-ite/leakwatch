"""End to end on freshly generated data: the checks find what was planted."""
import pandas as pd
import pytest

from leakage.generate import generate
from leakage.pipeline import run_demo
from leakage.score import miss_reasons, scorecard, traps_ignored


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    generate(n_invoices=5000, seed=11, data_dir=root, vendors_path=root / "vendors.csv")
    invoices, findings = run_demo(root, root / "vendors.csv")
    return invoices, findings, pd.read_csv(root / "answer_key.csv")


def test_only_the_indistinguishable_traps_are_flagged(result):
    _, findings, key = result
    traps = traps_ignored(findings, key).set_index("trap")
    # credit-noted duplicates must be left alone; recurring invoices (new number, same
    # amount) are a sanity check: neither rule compares different invoice numbers
    for trap in ["decoy", "recurring"]:
        assert traps.loc[trap, "total"] > 0 and traps.loc[trap, "ignored"] == traps.loc[trap, "total"]
    # a reused number a year later and a look-alike number cannot be told apart from
    # leakage by these rules: they are the known false positives
    for trap in ["reused_number", "lookalike_number"]:
        assert traps.loc[trap, "ignored"] == 0


def test_precision_is_honest(result):
    _, findings, key = result
    card = scorecard(findings, key).set_index("check")
    assert card.loc["price_overcharge", "precision"] == 1.0
    assert 0.8 <= card.loc["exact_duplicate", "precision"] < 1.0
    assert 0.8 <= card.loc["ocr_duplicate", "precision"] < 1.0


def test_catches_most_planted_leakage(result):
    _, findings, key = result
    card = scorecard(findings, key).set_index("check")
    assert card.loc["exact_duplicate", "recall"] >= 0.9
    assert card.loc["price_overcharge", "recall"] >= 0.9
    assert 0.7 <= card.loc["ocr_duplicate", "recall"] < 1.0


def test_every_miss_is_explained(result):
    invoices, findings, key = result
    reasons = miss_reasons(findings, key, invoices)
    assert "other" not in set(reasons["reason"])
    assert reasons["missed"].sum() == (scorecard(findings, key)["planted"]
                                       - scorecard(findings, key)["caught"]).sum()


def test_findings_have_vendor_names(result):
    _, findings, _ = result
    assert findings["vendor_name"].notna().all()


def test_detect_on_upload_without_contracts():
    from leakage.clean import prepare_upload
    from leakage.pipeline import detect
    user = prepare_upload(pd.DataFrame({
        "invoice_id": ["X1", "X2"], "vendor_id": ["V9", "V9"], "invoice_no": ["0012", "0O12"],
        "paid_date": ["2026-02-01", "2026-02-10"], "item": ["Widget", "Widget"], "qty": ["3", "3"],
        "unit_price": ["2.50", "2.50"], "amount": ["7.50", "7.50"], "doc_type": ["INV", "INV"]}))
    found = detect(user, pd.DataFrame(columns=["vendor_id", "item", "contract_price"]),
                   pd.DataFrame(columns=["vendor_id", "vendor_name"]))
    assert list(found["invoice_id"]) == ["X2"] and found.loc[0, "check_type"] == "ocr_duplicate"
