"""Check Tab 1's invoices against the terms extracted from contracts and emails."""
import pandas as pd

from contracts.generate_contracts import TERM_END
from contracts.terms import price_schedule


def revised_price_overcharges(invoices: pd.DataFrame, schedule: pd.DataFrame,
                              original: pd.DataFrame) -> pd.DataFrame:
    """Invoices billed above a price that an email lowered.

    Only the gap between the revised price and the original contract price is counted:
    anything billed above the original price is already a Tab 1 price_overcharge.
    """
    lines = (invoices[invoices["doc_type"] == "INV"]
             .merge(schedule, on=["vendor_id", "item"], suffixes=("", "_agreed"))
             .merge(original, on=["vendor_id", "item"]))
    in_period = (lines["paid_date"] >= lines["valid_from"]) & (lines["paid_date"] <= lines["valid_to"])
    lines = lines[in_period & (lines["unit_price_agreed"] < lines["contract_price"])].copy()
    lines["recoverable"] = ((lines[["unit_price", "contract_price"]].min(axis=1)
                             - lines["unit_price_agreed"]) * lines["qty"]).round(2)
    lines = lines[lines["recoverable"] > 0.005]
    lines["reason"] = [f"Billed {b:.2f} after the price was revised to {r:.2f} on {s}"
                       for b, r, s in zip(lines["unit_price"], lines["unit_price_agreed"], lines["valid_from"])]
    return lines[["invoice_id", "vendor_id", "item", "paid_date", "qty", "unit_price",
                  "contract_price", "unit_price_agreed", "recoverable", "reason"]].reset_index(drop=True)


def missed_rebates(invoices: pd.DataFrame, rebates: pd.DataFrame) -> pd.DataFrame:
    """For each rebate term: units and spend in its window, and the rebate due if the
    threshold was reached. The invoice data has no rebate credits, so a due rebate is missed."""
    rows = []
    for t in rebates.itertuples(index=False):
        end = t.effective_to if pd.notna(t.effective_to) else TERM_END.isoformat()
        window = invoices[(invoices["vendor_id"] == t.vendor_id) & (invoices["item"] == t.item)
                          & (invoices["doc_type"] == "INV")
                          & (invoices["paid_date"] >= t.effective_from) & (invoices["paid_date"] <= end)]
        qty, spend = int(window["qty"].sum()), float(window["amount"].sum())
        reached = qty >= t.rebate_threshold_qty
        rows.append({"vendor_id": t.vendor_id, "item": t.item, "window": f"{t.effective_from} to {end}",
                     "rebate_pct": t.rebate_pct, "threshold_qty": int(t.rebate_threshold_qty),
                     "qty_in_window": qty, "spend_in_window": round(spend, 2), "reached": reached,
                     "rebate_due": round(spend * t.rebate_pct / 100, 2) if reached else 0.0})
    return pd.DataFrame(rows)


def run_compliance(terms: pd.DataFrame, invoices: pd.DataFrame, original: pd.DataFrame,
                   already_claimed: set[str] = frozenset()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both checks from one table of terms (true terms, or trusted extracted terms).

    already_claimed: invoice ids Tab 1 recovers in full (duplicate payments). They are
    skipped by both checks: their money is already claimed, and a duplicate is the same
    units paid twice, not extra units towards a rebate threshold.
    """
    real = invoices[~invoices["invoice_id"].isin(already_claimed)]
    rebates = terms[terms["rebate_pct"].notna() & terms["rebate_threshold_qty"].notna()]
    return (revised_price_overcharges(real, price_schedule(terms), original),
            missed_rebates(real, rebates))
