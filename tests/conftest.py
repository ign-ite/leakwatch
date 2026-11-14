import pandas as pd
import pytest

from leakage.load import load_db

COLUMNS = ["invoice_id", "vendor_id", "vendor_needs_review", "invoice_no", "invoice_no_clean",
           "paid_date", "item", "qty", "unit_price", "amount", "doc_type"]


def invoice(invoice_id, invoice_no, paid_date, amount=100.0, vendor_id="V001", doc_type="INV",
            clean=None, item="Malt (kg)", qty=10, unit_price=10.0, review=0):
    return [invoice_id, vendor_id, review, invoice_no, clean or invoice_no.lstrip("0"),
            paid_date, item, qty, unit_price, amount, doc_type]


@pytest.fixture
def make_db():
    """make_db(rows, contracts=None) -> sqlite connection with invoices + contracts tables."""
    def _make(rows, contracts=None):
        invoices = pd.DataFrame(rows, columns=COLUMNS)
        contracts = contracts if contracts is not None else pd.DataFrame(
            {"vendor_id": ["V001"], "item": ["Malt (kg)"], "contract_price": [10.0]})
        return load_db(invoices, contracts)
    return _make
