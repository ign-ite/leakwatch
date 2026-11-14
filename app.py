"""Leakwatch: find money leaking out of accounts payable."""
from pathlib import Path

import pandas as pd
import streamlit as st

from leakage.clean import UPLOAD_COLUMNS, prepare_upload
from leakage.load import CHECKS, sql_text
from leakage.memo import build_memo
from leakage.pipeline import detect, run_demo
from leakage.score import decoys_ignored, scorecard, top_n_share, value_caught

ROOT = Path(__file__).parent
DATA = ROOT / "data"
VENDORS = ROOT / "shared" / "vendors.csv"
CHECK_LABELS = {"exact_duplicate": "Exact duplicate", "ocr_duplicate": "OCR duplicate",
                "price_overcharge": "Price overcharge"}

st.set_page_config(page_title="Leakwatch", layout="wide")


@st.cache_data
def demo():
    invoices, findings = run_demo(DATA, VENDORS)
    return invoices, findings, pd.read_csv(DATA / "answer_key.csv")


def inr(x: float) -> str:
    return f"₹{x:,.0f}"


def findings_table(findings: pd.DataFrame) -> None:
    types = st.multiselect("Leakage type", list(CHECK_LABELS), default=list(CHECK_LABELS),
                           format_func=CHECK_LABELS.get)
    vendor = st.selectbox("Vendor", ["All"] + sorted(findings["vendor_name"].dropna().unique()))
    shown = findings[findings["check_type"].isin(types)]
    if vendor != "All":
        shown = shown[shown["vendor_name"] == vendor]
    st.dataframe(
        shown[["invoice_id", "vendor_name", "check_type", "recoverable", "confidence", "priority", "reason"]]
        .rename(columns={"invoice_id": "Invoice", "vendor_name": "Vendor", "check_type": "Type",
                         "recoverable": "Recoverable (₹)", "confidence": "Confidence",
                         "priority": "Priority", "reason": "Why flagged"}),
        hide_index=True, width="stretch")


def leakage_tab() -> None:
    invoices, findings, key = demo()
    st.markdown(
        f"Three ERP exports ({len(invoices):,} invoices) with different date, number and vendor-name "
        "formats are cleaned into one table, loaded into SQLite and checked with three SQL rules. "
        "The data is synthetic with planted problems, so every check can be scored.")

    ignored, total_decoys = decoys_ignored(findings, key)
    c1, c2, c3 = st.columns(3)
    c1.metric("Recoverable value flagged", inr(findings["recoverable"].sum()))
    c2.metric("Planted leakage caught (by value)", f"{value_caught(findings, key):.0%}")
    c3.metric("Credit-note decoys correctly ignored", f"{ignored} / {total_decoys}")
    n = 20
    st.info(f"The top {n} claims by priority hold **{top_n_share(findings, n):.0%}** of recoverable "
            "value. Priority = recoverable value × confidence.")

    st.subheader("Findings")
    findings_table(findings)
    st.download_button("Download audit memo (markdown)", build_memo(findings),
                       file_name="audit_memo.md", mime="text/markdown")

    st.subheader("Scorecard vs answer key")
    st.dataframe(scorecard(findings, key), hide_index=True)
    st.caption("OCR misses are duplicates paid more than 30 days apart, outside the rule's window. "
               "Widening the window would catch them, at the cost of more false positives on real data.")
    unmatched = invoices["vendor_id"].isna()
    review = invoices["vendor_needs_review"] == 1
    st.caption(f"Vendor matching: {review.sum()} invoices matched fuzzily (confidence −0.2), "
               f"{unmatched.sum()} unmatched and excluded from checks.")

    st.subheader("The SQL")
    for name in CHECKS:
        with st.expander(f"Show {name}.sql"):
            st.code(sql_text(name), language="sql")

    st.subheader("Run the checks on your own file")
    st.caption("CSV columns: " + ", ".join(UPLOAD_COLUMNS) + ". doc_type is INV or CRN; "
               "dates are YYYY-MM-DD. Price checks need a contract price list, so uploads "
               "run the two duplicate checks.")
    upload = st.file_uploader("Upload CSV", type="csv")
    if upload is not None:
        try:
            user = prepare_upload(pd.read_csv(upload, dtype=str))
        except ValueError as err:
            st.error(str(err))
        else:
            no_contracts = pd.DataFrame(columns=["vendor_id", "item", "contract_price"])
            found = detect(user, no_contracts, pd.DataFrame(columns=["vendor_id", "vendor_name"]))
            st.write(f"{len(found)} findings worth {inr(found['recoverable'].sum())}")
            st.dataframe(found[["invoice_id", "check_type", "recoverable", "reason"]], hide_index=True)

    with st.expander("Limitations & what I'd do next"):
        st.markdown(
            "- Synthetic data is cleaner than real AP data; real exports have more formats and missing fields.\n"
            "- Multi-currency invoices and partial payments are not handled.\n"
            "- Confidence scores are hand-set; next step is learning them from confirmed recoveries.\n"
            "- Amounts must match exactly; next step is fuzzy amount matching (rounding, tax differences).")


st.title("Leakwatch")
st.caption("Finding money that leaks out of accounts payable: duplicate payments and overcharges.")
(tab1,) = st.tabs(["Invoice leakage"])
with tab1:
    leakage_tab()
