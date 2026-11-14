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


def decoys_ignored(findings: pd.DataFrame, answer_key: pd.DataFrame) -> tuple[int, int]:
    """(decoys not flagged, total decoys). Decoys are duplicates already reversed by a credit note."""
    decoys = set(answer_key.loc[answer_key["type"] == "decoy", "invoice_id"])
    return len(decoys - set(findings["invoice_id"])), len(decoys)


def value_caught(findings: pd.DataFrame, answer_key: pd.DataFrame) -> float:
    """Share of planted leakage value (INR) that was flagged."""
    real = answer_key[answer_key["type"] != "decoy"]
    hit = real["invoice_id"].isin(set(findings["invoice_id"]))
    return round(real.loc[hit, "amount"].sum() / real["amount"].sum(), 3)


def top_n_share(ranked: pd.DataFrame, n: int) -> float:
    """Share of total recoverable value in the top-n findings by priority."""
    total = ranked["recoverable"].sum()
    return round(ranked["recoverable"].head(n).sum() / total, 3) if total else 0.0
