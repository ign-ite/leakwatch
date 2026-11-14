import pandas as pd

from leakage.memo import build_memo
from leakage.score import add_priority


def test_memo_has_sections_totals_and_top_claims():
    findings = add_priority(pd.DataFrame({
        "invoice_id": ["B-1", "C-1", "A-1"],
        "vendor_id": ["V001", "V002", "V001"],
        "vendor_name": ["Konkan Agro Foods", "Bharat Tin Works", "Konkan Agro Foods"],
        "check_type": ["exact_duplicate", "ocr_duplicate", "price_overcharge"],
        "recoverable": [600.0, 300.0, 100.0],
        "vendor_needs_review": [0, 0, 0],
    }))
    memo = build_memo(findings, top_n=2)
    for heading in ["## Summary", "## By leakage type", "## Top vendors",
                    "## File these 2 claims first", "## Caveats"]:
        assert heading in memo
    assert "3 recovery claims worth **₹1,000**" in memo
    assert "cover **90%**" in memo                     # top 2 = 600 + 300 of 1000
    assert "| Konkan Agro Foods | ₹700 |" in memo       # vendor total
    assert "A-1" not in memo.split("## File these")[1].split("## Caveats")[0]
