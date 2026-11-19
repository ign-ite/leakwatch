import json

import pandas as pd
import pytest

from contracts.generate_contracts import generate_documents, window_qty
from leakage.generate import generate


@pytest.fixture(scope="module")
def docs(tmp_path_factory):
    root = tmp_path_factory.mktemp("docs")
    generate(n_invoices=3000, seed=7, data_dir=root, vendors_path=root / "vendors.csv")
    generate_documents(seed=1, data_dir=root, vendors_path=root / "vendors.csv")
    key = json.loads((root / "contract_answer_key.json").read_text())
    return root, key


def by_type(key, doc_type):
    return [d for d in key if d["type"] == doc_type]


def test_writes_one_text_file_per_document(docs):
    root, key = docs
    assert len(by_type(key, "contract")) == 30 and len(by_type(key, "email")) == 10
    for d in key:
        assert (root / "documents" / f"{d['doc_id']}.txt").read_text().strip()


def test_contract_prices_match_tab1_price_list(docs):
    root, key = docs
    price = pd.read_csv(root / "contracts.csv").set_index(["vendor_id", "item"])["contract_price"]
    terms = [t for d in by_type(key, "contract") for t in d["terms"]]
    assert len(terms) == 90
    for t in terms:
        assert t["unit_price"] == price[(t["vendor_id"], t["item"])]
        assert (t["effective_from"], t["effective_to"]) == ("2025-04-01", "2026-03-31")


def test_price_revisions_only_go_down(docs):
    root, key = docs
    price = pd.read_csv(root / "contracts.csv").set_index(["vendor_id", "item"])["contract_price"]
    revisions = [t for d in by_type(key, "email") for t in d["terms"] if t["unit_price"] is not None]
    assert len(revisions) == 6
    for t in revisions:
        assert t["unit_price"] < price[(t["vendor_id"], t["item"])]
        assert t["effective_to"] is None   # "until further notice"


def test_every_email_quotes_an_earlier_message(docs):
    root, key = docs
    for d in by_type(key, "email"):
        text = (root / "documents" / f"{d['doc_id']}.txt").read_text()
        assert "\n> " in text


def test_half_the_rebate_thresholds_are_reached(docs):
    root, key = docs
    from leakage.clean import clean_all
    from leakage.pipeline import read_erps
    invoices = clean_all(read_erps(root), pd.read_csv(root / "vendors.csv"))
    rebates = [t for d in by_type(key, "email") for t in d["terms"] if t["rebate_pct"] is not None]
    reached = [window_qty(invoices, t) >= t["rebate_threshold_qty"] for t in rebates]
    assert sorted(reached) == [False, False, True, True]


def test_same_seed_same_documents(tmp_path, docs):
    root, _ = docs
    for run in ("one", "two"):
        out = tmp_path / run
        generate_documents(seed=1, data_dir=root, vendors_path=root / "vendors.csv",
                           out_dir=out / "documents", key_path=out / "key.json")
    assert (tmp_path / "one" / "key.json").read_text() == (tmp_path / "two" / "key.json").read_text()
    assert (tmp_path / "one" / "documents" / "E01.txt").read_text() == \
        (tmp_path / "two" / "documents" / "E01.txt").read_text()
