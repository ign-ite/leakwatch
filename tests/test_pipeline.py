"""End to end on freshly generated data: the checks find what was planted."""
import pandas as pd
import pytest

from leakage.generate import generate
from leakage.pipeline import run_demo
from leakage.score import decoys_ignored, scorecard


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    generate(n_invoices=5000, seed=11, data_dir=root, vendors_path=root / "vendors.csv")
    invoices, findings = run_demo(root, root / "vendors.csv")
    return findings, pd.read_csv(root / "answer_key.csv")


def test_no_false_positives(result):
    findings, key = result
    assert (scorecard(findings, key)["precision"] == 1.0).all()


def test_catches_most_planted_leakage(result):
    findings, key = result
    card = scorecard(findings, key).set_index("check")
    assert card.loc["exact_duplicate", "recall"] >= 0.9
    assert card.loc["price_overcharge", "recall"] >= 0.9
    # every 10th OCR duplicate is planted >30 days out, beyond the rule's window
    assert 0.7 <= card.loc["ocr_duplicate", "recall"] < 1.0


def test_all_decoys_ignored(result):
    findings, key = result
    ignored, total = decoys_ignored(findings, key)
    assert total > 0 and ignored == total


def test_findings_have_vendor_names(result):
    findings, _ = result
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
