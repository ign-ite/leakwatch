"""Generate synthetic accounts-payable data from three fake ERPs, with planted leakage.

Every planted problem is written to answer_key.csv so detection can be scored.
Run: uv run python -m leakage.generate
"""
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

VENDOR_NAMES = [
    "Konkan Agro Foods", "Bharat Tin Works", "Sahyadri Polymers", "Malabar Glassware",
    "Udupi Print House", "Nandi Corrugators", "Kaveri Malt Mills", "Western Ghats Logistics",
    "Tulu Nadu Labels", "Coastal Shrink Films", "Deccan Crown Caps", "Mysore Pallet Co",
    "Netravati Packaging", "Kodagu Spice Traders", "Arabian Sea Freight", "Hampi Steel Drums",
    "Vijaya Bottling Supplies", "Shimoga Paper Board", "Karwar Marine Plastics", "Belur Cartons",
    "Gokarna Chemicals", "Agumbe Rubber Products", "Panambur Port Services", "Chikmagalur Estates",
    "Sringeri Wraps", "Kollur Industrial Gases", "Moodbidri Tooling", "Puttur Cashew Exports",
    "Kundapur Fisheries", "Bantwal Brick Kilns", "Surathkal Fabricators", "Ullal Cold Storage",
    "Hosur Auto Parts", "Tumkur Grain Traders", "Dharwad Textiles", "Gadag Printing Inks",
    "Raichur Cotton Bales", "Bidar Metalcraft", "Kolar Electricals", "Mandya Sugar Works",
]

# item -> typical unit price (INR)
ITEMS = {
    "PET bottle 500ml": 4.20, "Corrugated carton": 38.00, "Shrink film roll": 1250.00,
    "Crown caps (1000)": 310.00, "Printed labels (1000)": 450.00, "Malt (kg)": 52.00,
    "Pallet wrap": 890.00, "Glass bottle 650ml": 11.50,
}

START = date(2025, 4, 1)
ERPS = ["A", "B", "C"]

# share of base invoices used for each planted problem
PLANT_RATES = {"exact_duplicate": 0.008, "ocr_duplicate": 0.007,
               "price_overcharge": 0.005, "decoy": 0.002,
               "reused_number": 0.0005, "lookalike_number": 0.0005, "recurring": 0.001}
# Traps: legitimate payments that look like leakage. None of them is money lost.
TRAPS = ["decoy", "reused_number", "lookalike_number", "recurring"]


def make_vendors() -> pd.DataFrame:
    return pd.DataFrame({
        "vendor_id": [f"V{i:03d}" for i in range(1, len(VENDOR_NAMES) + 1)],
        "vendor_name": VENDOR_NAMES,
    })


def make_contracts(vendors: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    rows = []
    for vendor_id in vendors["vendor_id"]:
        for item in rng.sample(sorted(ITEMS), 3):
            price = round(ITEMS[item] * rng.uniform(0.9, 1.1), 2)
            rows.append({"vendor_id": vendor_id, "item": item, "contract_price": price})
    return pd.DataFrame(rows)


def vendor_variant(name: str, erp: str, rng: random.Random) -> str:
    """How each ERP spells a vendor name."""
    if erp == "A":
        return name
    if erp == "B":
        return name.upper() + " PVT LTD"
    r = rng.random()
    if r < 0.05:  # typo: one character dropped
        i = rng.randrange(1, len(name) - 1)
        return name[:i] + name[i + 1:] + " Private Limited"
    if r < 0.08:  # last word abbreviated: "Konkan Agro F."
        words = name.split()
        return " ".join(words[:-1] + [words[-1][0] + "."]) + " Private Limited"
    if r < 0.18:
        return name + " (India) Private Limited"
    return name + " Private Limited"


def corrupt_invoice_no(no: str, rng: random.Random) -> str:
    """Simulate one OCR misread: 0->O, 1->I, 5->S, or a dropped leading zero."""
    options = [(d, c) for d, c in (("0", "O"), ("1", "I"), ("5", "S")) if d in no]
    if no.startswith("0"):
        options.append(("drop", ""))
    digit, letter = rng.choice(options)
    if digit == "drop":
        return no[1:]
    positions = [i for i, ch in enumerate(no) if ch == digit]
    i = rng.choice(positions)
    return no[:i] + letter + no[i + 1:]


def _eu(x: float) -> str:
    """1234.5 -> '1.234,50'"""
    return f"{x:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _to_erp_frame(rows: list[dict], erp: str) -> pd.DataFrame:
    out = []
    for r in rows:
        d = r["paid_date"]
        if erp == "A":
            out.append({"doc_id": r["invoice_id"], "supplier": r["vendor_raw"],
                        "inv_no": r["invoice_no"], "pay_date": d.strftime("%d.%m.%Y"),
                        "material": r["item"], "quantity": r["qty"],
                        "price": _eu(r["unit_price"]), "total": _eu(r["amount"]),
                        "doc_type": r["doc_type"]})
        elif erp == "B":
            out.append({"ID": r["invoice_id"], "VENDOR_NAME": r["vendor_raw"],
                        "INVOICE_NUMBER": r["invoice_no"], "PAYMENT_DATE": d.isoformat(),
                        "ITEM": r["item"], "QTY": r["qty"],
                        "UNIT_PRICE": f"{r['unit_price']:.2f}", "AMOUNT": f"{r['amount']:.2f}",
                        "TYPE": r["doc_type"]})
        else:
            out.append({"record_id": r["invoice_id"], "vendor": r["vendor_raw"],
                        "invoice_no": r["invoice_no"], "paid_on": d.isoformat(),
                        "item": r["item"], "qty": r["qty"],
                        "unit_price": f"{r['unit_price']:.2f}", "amount": f"{r['amount']:.2f}",
                        "doc_type": r["doc_type"]})
    return pd.DataFrame(out)


def generate(n_invoices: int = 30_000, seed: int = 42,
             data_dir: Path = ROOT / "data", vendors_path: Path = ROOT / "shared" / "vendors.csv") -> None:
    rng = random.Random(seed)
    vendors = make_vendors()
    contracts = make_contracts(vendors, rng)
    names = dict(zip(vendors["vendor_id"], vendors["vendor_name"]))
    price = {(v, i): p for v, i, p in contracts.itertuples(index=False)}
    items_by_vendor = contracts.groupby("vendor_id")["item"].apply(list).to_dict()

    counters = {erp: 0 for erp in ERPS}
    next_no = {v: rng.randint(1_000, 90_000) for v in names}

    def new_row(erp: str, vendor_id: str, **fields) -> dict:
        counters[erp] += 1
        return {"invoice_id": f"{erp}-{counters[erp]:06d}", "erp": erp, "vendor_id": vendor_id,
                "vendor_raw": vendor_variant(names[vendor_id], erp, rng), **fields}

    # 1. Clean base invoices, all billed at the contract price.
    rows = []
    for _ in range(n_invoices):
        vendor_id = rng.choice(sorted(names))
        item = rng.choice(items_by_vendor[vendor_id])
        qty = rng.randint(10, 2000)
        unit = price[(vendor_id, item)]
        next_no[vendor_id] += rng.randint(1, 5)
        rows.append(new_row(rng.choice(ERPS), vendor_id,
                            invoice_no=f"{next_no[vendor_id]:07d}", item=item, qty=qty,
                            unit_price=unit, amount=round(qty * unit, 2), doc_type="INV",
                            paid_date=START + timedelta(days=rng.randrange(365))))

    # 2. Pick disjoint originals for each planted problem.
    counts = {k: max(1, round(n_invoices * r)) for k, r in PLANT_RATES.items()}
    picks = rng.sample(range(n_invoices), sum(counts.values()))
    answer, extra = [], []

    def copy_of(orig: dict, erp: str, days: int, invoice_no: str) -> dict:
        return new_row(erp, orig["vendor_id"], invoice_no=invoice_no, item=orig["item"],
                       qty=orig["qty"], unit_price=orig["unit_price"], amount=orig["amount"],
                       doc_type="INV", paid_date=orig["paid_date"] + timedelta(days=days))

    for kind, n in counts.items():
        for k in range(n):
            orig = rows[picks.pop()]
            other_erp = rng.choice([e for e in ERPS if e != orig["erp"]])
            if kind == "exact_duplicate":
                dup = copy_of(orig, other_erp, rng.randint(1, 60), orig["invoice_no"])
                extra.append(dup)
                answer.append((dup["invoice_id"], kind, dup["amount"], orig["invoice_id"]))
            elif kind == "ocr_duplicate":
                # every 10th one is paid >30 days later: the 30-day rule should miss it
                days = rng.randint(31, 60) if k % 10 == 0 else rng.randint(1, 30)
                dup = copy_of(orig, rng.choice(ERPS), days, corrupt_invoice_no(orig["invoice_no"], rng))
                extra.append(dup)
                answer.append((dup["invoice_id"], kind, dup["amount"], orig["invoice_id"]))
            elif kind == "price_overcharge":
                contract = orig["unit_price"]
                orig["unit_price"] = round(contract * rng.uniform(1.05, 1.25), 2)
                orig["amount"] = round(orig["qty"] * orig["unit_price"], 2)
                answer.append((orig["invoice_id"], kind,
                               round((orig["unit_price"] - contract) * orig["qty"], 2), orig["invoice_id"]))
            elif kind == "decoy":  # a duplicate that was already reversed by a credit note
                dup = copy_of(orig, other_erp, rng.randint(1, 60), orig["invoice_no"])
                credit = copy_of(orig, other_erp, 0, orig["invoice_no"])
                credit.update(doc_type="CRN", amount=-orig["amount"],
                              paid_date=dup["paid_date"] + timedelta(days=rng.randint(1, 15)))
                extra += [dup, credit]
                answer.append((dup["invoice_id"], kind, 0.0, orig["invoice_id"]))
            elif kind == "reused_number":
                # the vendor restarts numbering in a new year and bills the same amount again:
                # a real new invoice that the exact-duplicate rule cannot tell apart
                dup = copy_of(orig, rng.choice(ERPS), rng.randint(330, 400), orig["invoice_no"])
                extra.append(dup)
                answer.append((dup["invoice_id"], kind, 0.0, orig["invoice_id"]))
            elif kind == "lookalike_number":
                # a real, separate invoice whose number differs only by a look-alike
                # character (e.g. an 'S' series): without the document image it is
                # indistinguishable from a misread, which is why OCR flags carry 0.8 confidence
                dup = copy_of(orig, rng.choice(ERPS), rng.randint(1, 30),
                              corrupt_invoice_no(orig["invoice_no"], rng))
                extra.append(dup)
                answer.append((dup["invoice_id"], kind, 0.0, orig["invoice_id"]))
            else:  # recurring: same vendor, item and amount a month later, new invoice number
                dup = copy_of(orig, rng.choice(ERPS), rng.randint(25, 35),
                              f"{int(orig['invoice_no']) + 500_000:07d}")
                extra.append(dup)
                answer.append((dup["invoice_id"], kind, 0.0, orig["invoice_id"]))

    # 3. Write one file per ERP in that ERP's own format.
    data_dir.mkdir(parents=True, exist_ok=True)
    vendors_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = sorted(rows + extra, key=lambda r: r["paid_date"])
    for erp in ERPS:
        erp_rows = [r for r in all_rows if r["erp"] == erp]
        _to_erp_frame(erp_rows, erp).to_csv(data_dir / f"erp_{erp.lower()}.csv", index=False)
    vendors.to_csv(vendors_path, index=False)
    contracts.to_csv(data_dir / "contracts.csv", index=False)
    pd.DataFrame(answer, columns=["invoice_id", "type", "amount", "original_id"]).to_csv(
        data_dir / "answer_key.csv", index=False)


if __name__ == "__main__":
    generate()
    print("Wrote data/ and shared/vendors.csv")
