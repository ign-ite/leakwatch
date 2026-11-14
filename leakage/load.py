"""Load clean pandas tables into SQLite and run the detection SQL."""
import sqlite3
from pathlib import Path

import pandas as pd

SQL_DIR = Path(__file__).parent / "sql"
CHECKS = ["exact_duplicates", "ocr_duplicates", "price_overcharge"]


def load_db(invoices: pd.DataFrame, contracts: pd.DataFrame) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    invoices.to_sql("invoices", conn, index=False)
    contracts.to_sql("contracts", conn, index=False)
    return conn


def sql_text(name: str) -> str:
    return (SQL_DIR / f"{name}.sql").read_text()


def run_check(conn: sqlite3.Connection, name: str) -> pd.DataFrame:
    return pd.read_sql_query(sql_text(name), conn)


def run_all(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.concat([run_check(conn, name) for name in CHECKS], ignore_index=True)
