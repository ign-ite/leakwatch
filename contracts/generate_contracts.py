"""Generate synthetic supplier contracts and amendment emails, with an answer key.

Text is built from templates (not by an LLM), so the answer key is correct by construction.
Prices come from Tab 1's price list (data/contracts.csv). Rebate thresholds are set from
Tab 1's real invoice volumes, so some are reached and some are not.
Run after leakage.generate:  uv run python -m contracts.generate_contracts
"""
import json
import random
from datetime import date
from pathlib import Path

import pandas as pd

from leakage.clean import clean_all
from leakage.pipeline import read_erps

ROOT = Path(__file__).resolve().parent.parent
TERM_START, TERM_END = date(2025, 4, 1), date(2026, 3, 31)
BUYER = "Tidewater Beverages Pvt. Ltd."
PEOPLE = ["Anita Rao", "Suresh Shetty", "Meera Pai", "Rahul Kamath", "Divya Hegde", "Kiran Bhat"]

# How each catalogue item is written in prose. Extraction must map these back.
ITEM_PROSE = {
    "PET bottle 500ml": ["500 ml PET bottles", "PET bottles, 500ml"],
    "Corrugated carton": ["corrugated cartons", "corrugated shipping cartons"],
    "Shrink film roll": ["shrink film rolls", "rolls of shrink film"],
    "Crown caps (1000)": ["crown caps (per 1000)", "crown caps, packs of 1000"],
    "Printed labels (1000)": ["printed labels (per 1000)", "printed labels, packs of 1000"],
    "Malt (kg)": ["malt (per kg)", "malt, per kilogram"],
    "Pallet wrap": ["pallet wrap", "pallet wrap rolls"],
    "Glass bottle 650ml": ["650 ml glass bottles", "glass bottles, 650ml"],
}
N_CONTRACTS, N_REVISIONS, N_REBATES = 30, 6, 4


def _money(x: float, rng: random.Random) -> str:
    return rng.choice([f"₹{x:,.2f}", f"Rs. {x:,.2f}", f"INR {x:,.2f}", f"{x:,.2f} rupees"])


def _date(d: date, rng: random.Random) -> str:
    suffix = {1: "st", 2: "nd", 3: "rd", 21: "st", 22: "nd", 23: "rd", 31: "st"}.get(d.day, "th")
    return rng.choice([f"{d.day} {d:%B %Y}", f"{d:%d/%m/%Y}", f"{d:%B} {d.day}{suffix}, {d.year}"])


def _qty(q: int, rng: random.Random) -> str:
    return rng.choice([f"{q:,} units", f"{q / 1000:g}k units"])


def _term(vendor_id, item, start, end, unit_price=None, rebate_pct=None, threshold=None) -> dict:
    return {"vendor_id": vendor_id, "item": item, "unit_price": unit_price,
            "rebate_pct": rebate_pct, "rebate_threshold_qty": threshold,
            "effective_from": start.isoformat(), "effective_to": end.isoformat() if end else None}


def window_qty(invoices: pd.DataFrame, term: dict) -> int:
    """Units invoiced for the term's vendor + item between its effective dates."""
    end = term["effective_to"] or TERM_END.isoformat()
    rows = invoices[(invoices["vendor_id"] == term["vendor_id"]) & (invoices["item"] == term["item"])
                    & (invoices["doc_type"] == "INV")
                    & (invoices["paid_date"] >= term["effective_from"]) & (invoices["paid_date"] <= end)]
    return int(rows["qty"].sum())


def _contract(doc_id: str, vendor: str, lines: list[tuple[str, float]], rng: random.Random) -> str:
    title = rng.choice(["SUPPLY AGREEMENT", "RATE CONTRACT", "PRICE AGREEMENT"])
    price_lines = []
    for n, (item, price) in enumerate(lines, start=1):
        prose, money = rng.choice(ITEM_PROSE[item]), _money(price, rng)
        price_lines.append(rng.choice([
            f"1.{n} The Supplier shall supply {prose} at a unit price of {money} per unit.",
            f"1.{n} The agreed price for {prose} is {money} per unit.",
            f"1.{n} {prose[0].upper() + prose[1:]}: {money} per unit, fixed for the term.",
        ]))
    return "\n".join([
        title, f"Agreement No. {doc_id}/{TERM_START.year}", "",
        f"Buyer: {BUYER}", f"Supplier: {vendor} Pvt. Ltd.", "",
        f"This agreement is valid from {_date(TERM_START, rng)} to {_date(TERM_END, rng)}.", "",
        "1. Pricing", *price_lines, "",
        "2. Delivery and freight",
        "A freight surcharge of 2% applies to orders below 500 units.", "",
        "3. Payment",
        "Payment terms are net 45 days; overdue amounts attract interest at 1.5% per month.", "",
    ])


def _email(vendor: str, subject: str, body: str, quoted: str, rng: random.Random) -> str:
    person = rng.choice(PEOPLE)
    slug = vendor.lower().split()[0]
    earlier = _date(date(2025, rng.randint(5, 7), rng.randint(1, 28)), rng)
    return "\n".join([
        f"From: {person} <accounts@{slug}.example>", "To: procurement@tidewater.example",
        f"Subject: {subject}", "", "Hello,", "", body, "", "Regards,", person, f"{vendor} Pvt. Ltd.", "",
        f"> On {earlier}, procurement@tidewater.example wrote:", f"> {quoted}", "",
    ])


def generate_documents(seed: int = 7, data_dir: Path = ROOT / "data",
                       vendors_path: Path = ROOT / "shared" / "vendors.csv",
                       out_dir: Path | None = None, key_path: Path | None = None) -> None:
    rng = random.Random(seed)
    out_dir = out_dir or data_dir / "documents"
    key_path = key_path or data_dir / "contract_answer_key.json"
    vendors = pd.read_csv(vendors_path)
    names = dict(zip(vendors["vendor_id"], vendors["vendor_name"]))
    prices = pd.read_csv(data_dir / "contracts.csv")
    invoices = clean_all(read_erps(data_dir), vendors)

    docs, key = {}, []
    chosen = sorted(rng.sample(sorted(names), N_CONTRACTS))
    for n, vendor_id in enumerate(chosen, start=1):
        doc_id = f"C{n:02d}"
        rows = prices[prices["vendor_id"] == vendor_id]
        lines = list(zip(rows["item"], rows["contract_price"]))
        docs[doc_id] = _contract(doc_id, names[vendor_id], lines, rng)
        key.append({"doc_id": doc_id, "type": "contract", "terms": [
            _term(vendor_id, item, TERM_START, TERM_END, unit_price=price) for item, price in lines]})

    # Emails amend lines from the contracts above; each line is amended at most once.
    contract_lines = [(t["vendor_id"], t["item"], t["unit_price"]) for d in key for t in d["terms"]]
    picks = rng.sample(contract_lines, N_REVISIONS + N_REBATES)
    for n, (vendor_id, item, old) in enumerate(picks, start=1):
        doc_id, vendor, prose = f"E{n:02d}", names[vendor_id], rng.choice(ITEM_PROSE[item])
        if n <= N_REVISIONS:
            new = round(old * rng.uniform(0.88, 0.95), 2)
            start = date(2025, rng.randint(8, 12), 1)
            body = (f"Following our quarterly review, the price for {prose} will be revised to "
                    f"{_money(new, rng)} per unit, effective {_date(start, rng)}. "
                    "All other terms of our agreement remain unchanged.")
            quoted = f"Could you confirm the current price of {_money(old, rng)} per unit for {prose} still applies?"
            docs[doc_id] = _email(vendor, f"Revised pricing for {prose}", body, quoted, rng)
            key.append({"doc_id": doc_id, "type": "email",
                        "terms": [_term(vendor_id, item, start, None, unit_price=new)]})
        else:
            start, pct = date(2025, 10, 1), rng.choice([2.0, 3.0, 4.0, 5.0])
            term = _term(vendor_id, item, start, TERM_END, rebate_pct=pct)
            volume = window_qty(invoices, term)
            if n - N_REVISIONS <= N_REBATES // 2:   # first half: reachable at ~80% of real volume
                threshold = volume * 8 // 10 // 1000 * 1000
            else:                                   # second half: out of reach, above 130%
                threshold = (volume * 13 // 10 // 1000 + 1) * 1000
            term["rebate_threshold_qty"] = threshold
            body = (f"We are pleased to offer a volume rebate of {pct:g}% on {prose} once your purchases "
                    f"cross {_qty(term['rebate_threshold_qty'], rng)} between {_date(start, rng)} and "
                    f"{_date(TERM_END, rng)}. The rebate applies to all purchases of {prose} in that period.")
            quoted = f"Is there any volume discount on {prose}? We currently pay {_money(old, rng)} per unit."
            docs[doc_id] = _email(vendor, f"Volume rebate on {prose}", body, quoted, rng)
            key.append({"doc_id": doc_id, "type": "email", "terms": [term]})

    out_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, text in docs.items():
        (out_dir / f"{doc_id}.txt").write_text(text)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps(key, indent=2) + "\n")


if __name__ == "__main__":
    generate_documents()
    print("Wrote data/documents/ and data/contract_answer_key.json")
