import pandas as pd

from contracts.evaluate import (extracted_terms, field_accuracy, review_queue, truth_terms,
                                wrong_extractions)

VENDORS = pd.DataFrame({"vendor_id": ["V001"], "vendor_name": ["Konkan Agro Foods"]})

KEY = [
    {"doc_id": "C01", "type": "contract", "terms": [
        {"vendor_id": "V001", "item": "Malt (kg)", "unit_price": 52.4, "rebate_pct": None,
         "rebate_threshold_qty": None, "effective_from": "2025-04-01", "effective_to": "2026-03-31"},
        {"vendor_id": "V001", "item": "Pallet wrap", "unit_price": 900.0, "rebate_pct": None,
         "rebate_threshold_qty": None, "effective_from": "2025-04-01", "effective_to": "2026-03-31"}]},
    {"doc_id": "E01", "type": "email", "terms": [
        {"vendor_id": "V001", "item": "Malt (kg)", "unit_price": 48.0, "rebate_pct": None,
         "rebate_threshold_qty": None, "effective_from": "2025-08-01", "effective_to": None}]},
]


def term(item, unit_price, start="2025-04-01", end="2026-03-31", confidence=0.95, evidence=True):
    return {"vendor": "Konkan Agro Foods Pvt. Ltd.", "item": item, "unit_price": unit_price,
            "rebate_pct": None, "rebate_threshold_qty": None, "effective_from": start,
            "effective_to": end, "source": "contract", "confidence": confidence,
            "evidence_quote": "q", "evidence_ok": evidence}


EXTRACTIONS = {"model": "m", "documents": [
    {"doc_id": "C01", "status": "ok", "terms": [
        term("malt (per kg)", 52.4),
        term("pallet wrap rolls", 990.0),                         # wrong price
    ]},
    {"doc_id": "E01", "status": "ok", "terms": [
        term("malt, per kilogram", 48.0, "2025-01-08", None),     # read 01/08 as January
        term("malt (per kg)", 52.4, "2025-05-04", None, confidence=0.4, evidence=False),  # quoted old price
    ]},
]}


def test_field_accuracy_per_field():
    acc = field_accuracy(extracted_terms(EXTRACTIONS, VENDORS), truth_terms(KEY)).set_index("field")
    assert acc.loc["found", "accuracy"] == 1.0              # all 3 true terms were extracted
    assert acc.loc["unit_price", "accuracy"] == round(2 / 3, 3)
    assert acc.loc["effective_from", "accuracy"] == round(2 / 3, 3)
    assert acc.loc["effective_to", "accuracy"] == 1.0


def test_wrong_extractions_lists_each_error():
    wrong = wrong_extractions(extracted_terms(EXTRACTIONS, VENDORS), truth_terms(KEY))
    rows = set(zip(wrong["doc_id"], wrong["field"], wrong["problem"]))
    assert ("C01", "unit_price", "expected 900.0, got 990.0") in rows
    assert ("E01", "effective_from", "expected 2025-08-01, got 2025-01-08") in rows
    assert ("E01", "term", "not in the answer key (extra term)") in rows


def test_missing_term_counts_as_not_found():
    only_malt = {"model": "m", "documents": [{"doc_id": "C01", "status": "ok",
                                              "terms": [term("malt (per kg)", 52.4)]}]}
    acc = field_accuracy(extracted_terms(only_malt, VENDORS), truth_terms(KEY)).set_index("field")
    assert acc.loc["found", "accuracy"] == round(1 / 3, 3)


def test_review_queue_reasons():
    queue = review_queue(extracted_terms(EXTRACTIONS, VENDORS))
    assert len(queue) == 1
    assert queue.iloc[0]["doc_id"] == "E01"
    assert queue.iloc[0]["why"] == "low confidence (0.4); evidence quote not found in document"


def test_failed_documents_contribute_no_terms():
    failed = {"model": "m", "documents": [{"doc_id": "C01", "status": "failed", "terms": []}]}
    assert extracted_terms(failed, VENDORS).empty


def test_trusted_terms_drop_the_review_queue():
    from contracts.evaluate import trusted_terms
    trusted = trusted_terms(extracted_terms(EXTRACTIONS, VENDORS))
    assert len(trusted) == 3 and (trusted["confidence"] >= 0.7).all()


def test_fields_are_scored_only_where_the_truth_has_a_value():
    acc = field_accuracy(extracted_terms(EXTRACTIONS, VENDORS), truth_terms(KEY)).set_index("field")
    assert acc.loc["rebate_pct", "total"] == 0 and pd.isna(acc.loc["rebate_pct", "accuracy"])
    assert acc.loc["effective_to", "total"] == 2           # the email revision has no end date


def test_a_value_filled_in_where_the_truth_is_empty_is_counted():
    wrong_end = {"model": "m", "documents": [{"doc_id": "E01", "status": "ok",
                                              "terms": [term("malt (per kg)", 48.0, "2025-08-01", "2026-03-31")]}]}
    acc = field_accuracy(extracted_terms(wrong_end, VENDORS), truth_terms(KEY)).set_index("field")
    assert acc.loc["effective_to", "wrongly_filled"] == 1


def test_found_only_scores_just_the_terms_that_were_extracted():
    only_malt = {"model": "m", "documents": [{"doc_id": "C01", "status": "ok",
                                              "terms": [term("malt (per kg)", 52.4)]}]}
    acc = field_accuracy(extracted_terms(only_malt, VENDORS), truth_terms(KEY), found_only=True)
    assert acc.set_index("field").loc["unit_price", "accuracy"] == 1.0


def test_extra_terms_are_counted():
    from contracts.evaluate import extra_terms
    assert extra_terms(extracted_terms(EXTRACTIONS, VENDORS), truth_terms(KEY)) == 1


def test_loosely_matched_vendor_goes_to_review():
    loose = {"model": "m", "documents": [{"doc_id": "C01", "status": "ok",
                                          "terms": [term("malt (per kg)", 52.4) | {"vendor": "Konkan Agro"}]}]}
    queue = review_queue(extracted_terms(loose, VENDORS))
    assert queue.iloc[0]["why"].startswith("vendor name matched loosely")
