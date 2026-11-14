"""Rank findings and score them against the answer key."""
import pandas as pd

CONFIDENCE = {"exact_duplicate": 1.0, "ocr_duplicate": 0.8, "price_overcharge": 1.0}
REVIEW_PENALTY = 0.2  # vendor matched only fuzzily -> less sure it's the same vendor
CHECK_TYPES = list(CONFIDENCE)


def add_priority(findings: pd.DataFrame) -> pd.DataFrame:
    """confidence by check type, minus a penalty for fuzzy vendor matches;
    priority = recoverable x confidence. Sorted highest priority first."""
    out = findings.copy()
    out["confidence"] = (out["check_type"].map(CONFIDENCE)
                         - REVIEW_PENALTY * out["vendor_needs_review"]).round(2)
    out["priority"] = (out["recoverable"] * out["confidence"]).round(2)
    return out.sort_values("priority", ascending=False, ignore_index=True)


def scorecard(findings: pd.DataFrame, answer_key: pd.DataFrame) -> pd.DataFrame:
    """Precision and recall per check type.
    precision = share of flags that are real; recall = share of real problems flagged."""
    rows = []
    for check in CHECK_TYPES:
        flagged = set(findings.loc[findings["check_type"] == check, "invoice_id"])
        planted = set(answer_key.loc[answer_key["type"] == check, "invoice_id"])
        caught = flagged & planted
        rows.append({
            "check": check, "planted": len(planted), "flagged": len(flagged), "caught": len(caught),
            "precision": round(len(caught) / len(flagged), 3) if flagged else None,
            "recall": round(len(caught) / len(planted), 3) if planted else None,
        })
    return pd.DataFrame(rows)


def traps_ignored(findings: pd.DataFrame, answer_key: pd.DataFrame) -> pd.DataFrame:
    """Per trap type (legitimate payments that look like leakage): how many were left unflagged."""
    flagged = set(findings["invoice_id"])
    traps = answer_key[~answer_key["type"].isin(CHECK_TYPES)]
    return (traps.assign(ignored=~traps["invoice_id"].isin(flagged))
            .groupby("type", sort=False)["ignored"].agg(ignored="sum", total="count")
            .rename_axis("trap").reset_index())


def _hits(findings: pd.DataFrame, planted: pd.DataFrame) -> pd.Series:
    """True where a planted problem was flagged by the check for its own type."""
    flagged = set(zip(findings["invoice_id"], findings["check_type"]))
    return pd.Series([pair in flagged for pair in zip(planted["invoice_id"], planted["type"])],
                     index=planted.index, dtype=bool)


def value_caught(findings: pd.DataFrame, answer_key: pd.DataFrame) -> float:
    """Share of planted leakage value (INR) flagged by the right check."""
    real = answer_key[answer_key["type"].isin(CHECK_TYPES)]
    return round(real.loc[_hits(findings, real), "amount"].sum() / real["amount"].sum(), 3)


def miss_reasons(findings: pd.DataFrame, answer_key: pd.DataFrame,
                 invoices: pd.DataFrame) -> pd.DataFrame:
    """Why each planted problem was missed: (check, reason, missed count)."""
    real = answer_key[answer_key["type"].isin(CHECK_TYPES)]
    missed = real[~_hits(findings, real)]
    info = invoices.set_index("invoice_id")[["vendor_id", "paid_date"]]
    rows = []
    for m in missed.itertuples(index=False):
        dup, orig = info.loc[m.invoice_id], info.loc[m.original_id]
        gap = (pd.Timestamp(dup["paid_date"]) - pd.Timestamp(orig["paid_date"])).days
        if pd.isna(dup["vendor_id"]) or pd.isna(orig["vendor_id"]):
            reason = "vendor name unmatched"
        elif m.type == "ocr_duplicate" and gap > 30:
            reason = "paid more than 30 days after the original"
        else:
            reason = "other"
        rows.append((m.type, reason))
    return (pd.DataFrame(rows, columns=["check", "reason"])
            .value_counts().rename("missed").reset_index())


def top_n_share(ranked: pd.DataFrame, n: int) -> float:
    """Share of total recoverable value in the top-n findings by priority."""
    total = ranked["recoverable"].sum()
    return round(ranked["recoverable"].head(n).sum() / total, 3) if total else 0.0
