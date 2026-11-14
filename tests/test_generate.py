import pandas as pd
import pytest

from leakage.clean import clean_invoice_no, match_vendor, normalize_vendor
from leakage.generate import VENDOR_NAMES, corrupt_invoice_no, generate, vendor_variant
import random


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("gen")
    generate(n_invoices=3000, seed=7, data_dir=root / "data", vendors_path=root / "vendors.csv")
    return root


def test_writes_all_files(generated):
    for name in ["erp_a.csv", "erp_b.csv", "erp_c.csv", "contracts.csv", "answer_key.csv"]:
        assert (generated / "data" / name).exists()
    assert len(pd.read_csv(generated / "vendors.csv")) == 40


def test_answer_key_ids_exist_in_erp_files(generated):
    d = generated / "data"
    ids = set(pd.read_csv(d / "erp_a.csv", dtype=str)["doc_id"]) \
        | set(pd.read_csv(d / "erp_b.csv", dtype=str)["ID"]) \
        | set(pd.read_csv(d / "erp_c.csv", dtype=str)["record_id"])
    key = pd.read_csv(d / "answer_key.csv")
    assert set(key["invoice_id"]) <= ids
    assert set(key["original_id"]) <= ids
    assert set(key["type"]) == {"exact_duplicate", "ocr_duplicate", "price_overcharge",
                                "decoy", "reused_number", "lookalike_number", "recurring"}


def test_same_seed_same_data(tmp_path):
    for run in ("one", "two"):
        generate(n_invoices=200, seed=3, data_dir=tmp_path / run, vendors_path=tmp_path / run / "v.csv")
    assert (tmp_path / "one" / "erp_c.csv").read_text() == (tmp_path / "two" / "erp_c.csv").read_text()


def test_corrupted_numbers_clean_back_to_original():
    rng = random.Random(0)
    for n in range(1000, 1500):
        original = f"{n:07d}"
        corrupted = corrupt_invoice_no(original, rng)
        assert corrupted != original
        assert clean_invoice_no(corrupted) == clean_invoice_no(original)


def test_vendor_variants_never_match_the_wrong_vendor():
    known = {normalize_vendor(n): i for i, n in enumerate(VENDOR_NAMES)}
    rng = random.Random(0)
    for i, name in enumerate(VENDOR_NAMES):
        for _ in range(50):
            for erp in "ABC":
                vendor_id, _ = match_vendor(vendor_variant(name, erp, rng), known)
                assert vendor_id in (i, None)
