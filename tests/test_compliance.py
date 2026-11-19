import pandas as pd

from contracts.compliance import missed_rebates, revised_price_overcharges


def inv(invoice_id, paid_date, unit_price, qty=10, item="Malt (kg)", vendor_id="V001", doc_type="INV"):
    return {"invoice_id": invoice_id, "vendor_id": vendor_id, "item": item, "paid_date": paid_date,
            "qty": qty, "unit_price": unit_price, "amount": round(qty * unit_price, 2),
            "doc_type": doc_type}


ORIGINAL = pd.DataFrame({"vendor_id": ["V001"], "item": ["Malt (kg)"], "contract_price": [100.0]})
SCHEDULE = pd.DataFrame([  # contract 100, revised to 90 from 1 Aug
    {"vendor_id": "V001", "item": "Malt (kg)", "unit_price": 100.0,
     "valid_from": "2025-04-01", "valid_to": "2025-07-31"},
    {"vendor_id": "V001", "item": "Malt (kg)", "unit_price": 90.0,
     "valid_from": "2025-08-01", "valid_to": "2026-03-31"},
])


def test_revised_price_overcharge_only_after_effective_date():
    invoices = pd.DataFrame([
        inv("A-1", "2025-07-31", 100.0),             # before revision: fine
        inv("A-2", "2025-08-01", 100.0),             # old price after revision: 10 x 10 = 100
        inv("A-3", "2025-09-15", 90.0),              # new price: fine
        inv("A-4", "2025-09-15", 100.0, doc_type="CRN"),
    ])
    out = revised_price_overcharges(invoices, SCHEDULE, ORIGINAL)
    assert out[["invoice_id", "recoverable"]].values.tolist() == [["A-2", 100.0]]


def test_revised_price_overcharge_excludes_what_tab1_already_counts():
    # billed 120 vs original 100 (Tab 1 claims 20/unit) vs revised 90: only 10/unit is new
    out = revised_price_overcharges(pd.DataFrame([inv("A-1", "2025-10-01", 120.0)]), SCHEDULE, ORIGINAL)
    assert out.loc[0, "recoverable"] == 100.0


def test_revised_price_overcharge_ignores_other_vendors_and_items():
    invoices = pd.DataFrame([inv("A-1", "2025-10-01", 100.0, vendor_id="V002"),
                             inv("A-2", "2025-10-01", 100.0, item="Pallet wrap")])
    assert revised_price_overcharges(invoices, SCHEDULE, ORIGINAL).empty


REBATE = pd.DataFrame([{"vendor_id": "V001", "item": "Malt (kg)", "rebate_pct": 3.0,
                        "rebate_threshold_qty": 25, "effective_from": "2025-10-01",
                        "effective_to": "2026-03-31"}])


def test_missed_rebate_when_threshold_reached_in_window():
    invoices = pd.DataFrame([
        inv("A-1", "2025-09-30", 100.0, qty=50),     # before window: doesn't count
        inv("A-2", "2025-10-01", 100.0, qty=10),
        inv("A-3", "2026-03-31", 100.0, qty=15),     # 25 units in window: reached
    ])
    row = missed_rebates(invoices, REBATE).iloc[0]
    assert row["qty_in_window"] == 25 and bool(row["reached"])
    assert row["rebate_due"] == 75.0                 # 3% of 2,500 spend in window


def test_rebate_below_threshold_is_not_due():
    row = missed_rebates(pd.DataFrame([inv("A-1", "2025-11-01", 100.0, qty=24)]), REBATE).iloc[0]
    assert not row["reached"] and row["rebate_due"] == 0.0


def test_run_compliance_uses_prices_and_rebates_from_one_term_table():
    from contracts.compliance import run_compliance
    terms = pd.DataFrame([
        {"vendor_id": "V001", "item": "Malt (kg)", "unit_price": 100.0, "rebate_pct": None,
         "rebate_threshold_qty": None, "effective_from": "2025-04-01", "effective_to": "2026-03-31"},
        {"vendor_id": "V001", "item": "Malt (kg)", "unit_price": 90.0, "rebate_pct": None,
         "rebate_threshold_qty": None, "effective_from": "2025-08-01", "effective_to": None},
        REBATE.iloc[0].to_dict() | {"unit_price": None},
    ])
    invoices = pd.DataFrame([inv("A-1", "2025-10-01", 100.0, qty=30)])
    overcharges, rebates = run_compliance(terms, invoices, ORIGINAL)
    assert overcharges["recoverable"].tolist() == [300.0]
    assert rebates["rebate_due"].tolist() == [90.0]


def test_run_compliance_skips_invoices_tab1_already_claims_in_full():
    from contracts.compliance import run_compliance
    terms = pd.DataFrame([
        {"vendor_id": "V001", "item": "Malt (kg)", "unit_price": 100.0, "rebate_pct": None,
         "rebate_threshold_qty": None, "effective_from": "2025-04-01", "effective_to": "2026-03-31"},
        {"vendor_id": "V001", "item": "Malt (kg)", "unit_price": 90.0, "rebate_pct": None,
         "rebate_threshold_qty": None, "effective_from": "2025-08-01", "effective_to": None},
    ])
    invoices = pd.DataFrame([inv("A-1", "2025-10-01", 100.0), inv("B-1", "2025-10-05", 100.0)])
    overcharges, _ = run_compliance(terms, invoices, ORIGINAL, already_claimed={"B-1"})
    assert overcharges["invoice_id"].tolist() == ["A-1"]


def test_missed_rebates_with_no_terms_still_has_columns():
    out = missed_rebates(pd.DataFrame([inv("A-1", "2025-11-01", 100.0)]), REBATE.iloc[0:0])
    assert out.empty and "rebate_due" in out.columns


def test_a_rebate_extracted_twice_is_claimed_once():
    from contracts.compliance import run_compliance
    rebate = REBATE.iloc[0].to_dict() | {"unit_price": None}
    _, rebates = run_compliance(pd.DataFrame([rebate, rebate]),
                                pd.DataFrame([inv("A-1", "2025-10-01", 100.0, qty=30)]), ORIGINAL)
    assert rebates["rebate_due"].tolist() == [90.0]
