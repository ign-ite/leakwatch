"""Leakwatch: find money leaking out of accounts payable."""
import html
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from contracts.compliance import run_compliance
from contracts.evaluate import (FIELDS, extra_terms, extracted_terms, field_accuracy,
                                review_queue, trusted_terms, truth_terms, wrong_extractions)
from leakage.clean import UPLOAD_COLUMNS, prepare_upload
from leakage.load import CHECKS, sql_text
from leakage.memo import build_memo, inr
from leakage.pipeline import detect, run_demo
from leakage.score import (REVIEW_PENALTY, miss_reasons, scorecard, top_n_share, traps_ignored,
                           value_caught)

ROOT = Path(__file__).parent
DATA = ROOT / "data"
VENDORS = ROOT / "shared" / "vendors.csv"
EXTRACTIONS = Path(os.environ.get("LEAKWATCH_EXTRACTIONS", ROOT / "contracts" / "extractions.json"))
CHECK_LABELS = {"exact_duplicate": "Exact duplicate", "ocr_duplicate": "OCR duplicate",
                "price_overcharge": "Price overcharge"}

st.set_page_config(page_title="Leakwatch", layout="wide")


@st.cache_data
def demo():
    invoices, findings = run_demo(DATA, VENDORS)
    return invoices, findings, pd.read_csv(DATA / "answer_key.csv")


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

    traps = traps_ignored(findings, key)
    c1, c2, c3 = st.columns(3)
    c1.metric("Recoverable value flagged", inr(findings["recoverable"].sum()))
    c2.metric("Planted leakage caught (by value)", f"{value_caught(findings, key):.0%}")
    c3.metric("Legitimate look-alikes correctly ignored",
              f"{traps['ignored'].sum()} / {traps['total'].sum()}")
    n = 20
    st.info(f"The top {n} claims by priority hold **{top_n_share(findings, n):.0%}** of recoverable "
            "value. Priority = recoverable value × confidence.")

    st.subheader("Findings")
    findings_table(findings)
    st.download_button("Download audit memo (markdown)", build_memo(findings),
                       file_name="audit_memo.md", mime="text/markdown")

    st.subheader("Scorecard vs answer key")
    st.dataframe(scorecard(findings, key), hide_index=True)
    left, right = st.columns(2)
    left.markdown("**Why planted leakage was missed**")
    left.dataframe(miss_reasons(findings, key, invoices), hide_index=True)
    left.caption("Duplicates paid more than 30 days after the original fall outside the OCR rule's "
                 "window; widening it catches them but flags more legitimate repeat invoices. "
                 "Unmatched vendor names score below 0.75 and are left out of every check.")
    right.markdown("**Legitimate payments that look like leakage**")
    right.dataframe(traps, hide_index=True)
    right.caption("decoy: a duplicate already reversed by a credit note. recurring: same amount a "
                  "month later under a new number. reused_number: the vendor reused an invoice "
                  "number a year later. lookalike_number: a separate invoice whose number differs "
                  "only by O/0, I/1, S/5 or a leading zero. The last two are the false positives: telling them apart "
                  "needs a time limit on exact matches, or the document image.")
    unmatched = invoices["vendor_id"].isna()
    review = invoices["vendor_needs_review"] == 1
    st.caption(f"Vendor matching: {review.sum()} invoices matched fuzzily "
               f"(confidence −{REVIEW_PENALTY}), {unmatched.sum()} unmatched and excluded from checks.")

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
            "- Amounts must match exactly; next step is fuzzy amount matching (rounding, tax differences).\n"
            "- Exact matches have no time limit, so a vendor reusing a number a year later is flagged; "
            "next step is limiting matches to a window or fiscal year.")


def highlighted(text: str, quotes: list[str]) -> str:
    """Document text as HTML with each evidence quote that appears verbatim marked."""
    out = html.escape(text)
    for q in sorted(set(quotes), key=len, reverse=True):
        if q.strip() and q in text:
            out = out.replace(html.escape(q), f"<mark>{html.escape(q)}</mark>")
    return f'<pre style="white-space: pre-wrap; font-size: 0.85rem">{out}</pre>'


def contracts_tab() -> None:
    invoices, findings, _ = demo()
    extractions = json.loads(EXTRACTIONS.read_text())
    vendors = pd.read_csv(VENDORS)
    key = json.loads((DATA / "contract_answer_key.json").read_text())
    extracted, truth = extracted_terms(extractions, vendors), truth_terms(key)
    docs = {d["doc_id"]: d for d in extractions["documents"]}
    st.markdown(
        f"{len(docs)} supplier documents (rate contracts and emails that revise prices or offer "
        f"volume rebates) are read by **{extractions['model']}**, which returns each pricing or "
        "rebate term with a quote from the document as evidence. A term whose quote can't be found "
        "verbatim is not trusted. Only trusted terms are checked against the invoices from the first tab.")

    accuracy = field_accuracy(extracted, truth).set_index("field")
    found_terms = field_accuracy(extracted, truth, found_only=True).set_index("field").loc[FIELDS]
    fields_right = found_terms["correct"].sum() / found_terms["total"].sum() if found_terms["total"].sum() else 0
    failed = [d for d in docs.values() if d["status"] == "failed"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Terms found", f"{accuracy.loc['found', 'accuracy']:.0%}")
    c2.metric("Fields exactly right", f"{fields_right:.0%}",
              help="Over the terms that were found, counting only fields the document states.")
    c3.metric("Extra terms", extra_terms(extracted, truth),
              help="Extracted terms that aren't in the document's answer key: hallucinations, or "
                   "traps such as a quoted old price or a freight surcharge.")
    c4.metric("Documents that failed", f"{len(failed)} of {len(docs)}")
    if failed:
        st.warning("Failed: " + "; ".join(f"{d['doc_id']} ({d['error']})" for d in failed))

    st.subheader("Accuracy per field")
    st.dataframe(accuracy.reset_index(), hide_index=True)
    st.caption("Each field is scored only over terms whose document states it; wrongly_filled counts "
               "values the model added where the document states none (e.g. an end date it invented).")
    with st.expander("Every wrong extraction"):
        st.dataframe(wrong_extractions(extracted, truth), hide_index=True, width="stretch")

    st.subheader("Review queue")
    st.caption("Low confidence (< 0.7), an evidence quote not in the document, an unknown vendor "
               "or item, or a missing price or date. These are not used for claims.")
    queue = review_queue(extracted)
    st.dataframe(queue[["doc_id", "vendor", "item", "unit_price", "rebate_pct", "confidence",
                        "evidence_quote", "why"]], hide_index=True, width="stretch")

    st.subheader("Money the first tab can't see")
    claimed = set(findings.loc[findings["check_type"].isin(["exact_duplicate", "ocr_duplicate"]),
                               "invoice_id"])
    original = pd.read_csv(DATA / "contracts.csv")
    over, rebates = run_compliance(trusted_terms(extracted), invoices, original, claimed)
    true_over, true_rebates = run_compliance(truth, invoices, original, claimed)
    found_value = over["recoverable"].sum() + rebates["rebate_due"].sum()
    true_value = true_over["recoverable"].sum() + true_rebates["rebate_due"].sum()
    c1, c2 = st.columns(2)
    c1.metric("Recoverable with extracted terms", inr(found_value))
    c2.metric("Recoverable with the true terms", inr(true_value))
    st.caption("The gap is what extraction errors and review-queue holds cost. Overcharges count only "
               "the part below the original contract price (anything above is a first-tab finding); "
               "duplicate payments, already claimed in full, are skipped.")
    left, right = st.columns(2)
    left.markdown("**Billed above a revised price**")
    if len(over):
        left.dataframe(over.groupby(["vendor_id", "item"])["recoverable"].agg(["count", "sum"])
                       .round(0).reset_index(), hide_index=True)
    right.markdown("**Volume rebates**")
    if len(rebates):
        right.dataframe(rebates[["vendor_id", "item", "threshold_qty", "qty_in_window", "reached",
                                 "rebate_due"]], hide_index=True)

    st.subheader("Look at a document")
    doc_id = st.selectbox("Document", list(docs))
    doc = docs[doc_id]
    text = (DATA / "documents" / f"{doc_id}.txt").read_text()
    left, right = st.columns(2)
    left.markdown(highlighted(text, [t["evidence_quote"] for t in doc["terms"]]), unsafe_allow_html=True)
    if doc["terms"]:
        right.dataframe(pd.DataFrame(doc["terms"])[["item", "unit_price", "rebate_pct",
                        "rebate_threshold_qty", "effective_from", "effective_to", "confidence",
                        "evidence_ok"]], hide_index=True)
    else:
        right.info(f"No terms: {doc['error']}")

    st.subheader("Cost and speed")
    usage = pd.DataFrame(docs.values())[["doc_id", "status", "attempts", "prompt_tokens",
                                         "output_tokens", "latency_s"]]
    st.caption(f"{usage['prompt_tokens'].sum():,} input and {usage['output_tokens'].sum():,} output "
               f"tokens in total; median {usage['latency_s'].median():.1f} s per document.")
    with st.expander("Per document"):
        st.dataframe(usage, hide_index=True)


st.title("Leakwatch")
st.caption("Finding money that leaks out of accounts payable: duplicate payments, "
           "overcharges and contract terms that were never applied.")
if EXTRACTIONS.exists():
    tab1, tab2 = st.tabs(["Invoice leakage", "Contract terms"])
    with tab2:
        contracts_tab()
else:
    (tab1,) = st.tabs(["Invoice leakage"])
with tab1:
    leakage_tab()
