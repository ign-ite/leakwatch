"""End to end: raw ERP files -> ranked findings."""
from contextlib import closing
from pathlib import Path

import pandas as pd

from leakage.clean import clean_all
from leakage.load import load_db, run_all
from leakage.score import add_priority


def read_erps(data_dir: Path) -> dict[str, pd.DataFrame]:
    # dtype=str keeps invoice numbers like '0045123' intact (no int conversion).
    return {erp: pd.read_csv(data_dir / f"erp_{erp.lower()}.csv", dtype=str) for erp in "ABC"}


def detect(invoices: pd.DataFrame, contracts: pd.DataFrame, vendors: pd.DataFrame) -> pd.DataFrame:
    with closing(load_db(invoices, contracts)) as conn:
        findings = add_priority(run_all(conn))
    return findings.merge(vendors, on="vendor_id", how="left")


def run_demo(data_dir: Path, vendors_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (clean invoices, ranked findings) for the bundled synthetic data."""
    vendors = pd.read_csv(vendors_path)
    invoices = clean_all(read_erps(data_dir), vendors)
    contracts = pd.read_csv(data_dir / "contracts.csv")
    return invoices, detect(invoices, contracts, vendors)
