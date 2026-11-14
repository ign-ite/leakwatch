"""One-page markdown audit memo."""
import pandas as pd

CAVEATS = [
    "Findings are candidates for recovery, not confirmed overpayments; each claim needs the source documents.",
    "Vendor names matched fuzzily are marked for review and carry lower confidence.",
    "Multi-currency invoices and partial payments are not handled.",
]


def _inr(x: float) -> str:
    return f"₹{x:,.0f}"


def _table(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def build_memo(findings: pd.DataFrame, top_n: int = 10) -> str:
    """findings: ranked output of pipeline.detect (has vendor_name, priority)."""
    total = findings["recoverable"].sum()
    by_type = (findings.groupby("check_type")["recoverable"].agg(["count", "sum"])
               .sort_values("sum", ascending=False).reset_index())
    by_type["sum"] = by_type["sum"].map(_inr)
    by_type.columns = ["Leakage type", "Claims", "Recoverable"]
    by_vendor = (findings.groupby("vendor_name")["recoverable"].sum()
                 .sort_values(ascending=False).head(5).reset_index())
    by_vendor["recoverable"] = by_vendor["recoverable"].map(_inr)
    by_vendor.columns = ["Vendor", "Recoverable"]
    top = findings.head(top_n)[["invoice_id", "vendor_name", "check_type", "recoverable", "confidence"]].copy()
    top["recoverable"] = top["recoverable"].map(_inr)
    top.columns = ["Invoice", "Vendor", "Type", "Recoverable", "Confidence"]
    top_share = findings["recoverable"].head(top_n).sum() / total if total else 0
    return "\n\n".join([
        "# Accounts Payable Leakage — Audit Memo",
        "## Summary",
        f"{len(findings)} recovery claims worth **{_inr(total)}**. "
        f"The top {top_n} claims by priority cover **{top_share:.0%}** of that value.",
        "## By leakage type", _table(by_type),
        "## Top vendors", _table(by_vendor),
        f"## File these {top_n} claims first", _table(top),
        "## Caveats", "\n".join(f"- {c}" for c in CAVEATS),
    ])
