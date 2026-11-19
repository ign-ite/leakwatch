"""Score extracted terms against the answer key, and pick out terms a human should check."""
import pandas as pd

from contracts.terms import to_ids

FIELDS = ["unit_price", "rebate_pct", "rebate_threshold_qty", "effective_from", "effective_to"]
KEY_COLUMNS = ["doc_id", "vendor_id", "item"]
LOW_CONFIDENCE = 0.7
_TERM_COLUMNS = ["doc_id", "vendor", "item", *FIELDS, "source", "confidence",
                 "evidence_quote", "evidence_ok"]


def extracted_terms(extractions: dict, vendors: pd.DataFrame) -> pd.DataFrame:
    """One row per extracted term from documents that didn't fail, with Tab 1 ids."""
    rows = [{"doc_id": d["doc_id"], **t} for d in extractions["documents"]
            if d["status"] == "ok" for t in d["terms"]]
    return to_ids(pd.DataFrame(rows, columns=_TERM_COLUMNS), vendors)


def truth_terms(answer_key: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"doc_id": d["doc_id"], **t} for d in answer_key for t in d["terms"]])


def _same(a, b) -> bool:
    if pd.isna(a) or pd.isna(b):
        return pd.isna(a) and pd.isna(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 0.005
    return a == b


def _match(extracted: pd.DataFrame, truth: pd.DataFrame) -> tuple[list, list]:
    """Pair each true term with the extracted term (same doc, vendor, item) that agrees on
    the most fields. Returns (pairs of (truth row, extracted row or None), unmatched extras)."""
    pool = {i: row for i, row in extracted.iterrows()}
    pairs = []
    for _, t in truth.iterrows():
        candidates = [i for i, e in pool.items() if all(_same(e[c], t[c]) for c in KEY_COLUMNS)]
        best = max(candidates, key=lambda i: sum(_same(pool[i][f], t[f]) for f in FIELDS), default=None)
        pairs.append((t, pool.pop(best) if best is not None else None))
    return pairs, list(pool.values())


def field_accuracy(extracted: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Share of true terms found, then share with each field exactly right (missing = wrong)."""
    pairs, _ = _match(extracted, truth)
    rows = [("found", sum(e is not None for _, e in pairs))]
    rows += [(f, sum(e is not None and _same(e[f], t[f]) for t, e in pairs)) for f in FIELDS]
    return pd.DataFrame([{"field": f, "correct": c, "total": len(pairs),
                          "accuracy": round(c / len(pairs), 3)} for f, c in rows])


def wrong_extractions(extracted: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Every error: a wrong field, a term not extracted, or an extracted term that isn't real."""
    pairs, extras = _match(extracted, truth)
    rows = []
    for t, e in pairs:
        if e is None:
            rows.append((t["doc_id"], t["vendor_id"], t["item"], "term", "not extracted"))
            continue
        rows += [(t["doc_id"], t["vendor_id"], t["item"], f, f"expected {t[f]}, got {e[f]}")
                 for f in FIELDS if not _same(e[f], t[f])]
    rows += [(e["doc_id"], e["vendor_id"], e["item"], "term", "not in the answer key (extra term)")
             for e in extras]
    return pd.DataFrame(rows, columns=["doc_id", "vendor_id", "item", "field", "problem"])


def _review_reasons(extracted: pd.DataFrame) -> pd.Series:
    """Why each term needs a human check ("" if it doesn't)."""
    def why(t) -> str:
        reasons = []
        if t["confidence"] < LOW_CONFIDENCE:
            reasons.append(f"low confidence ({t['confidence']:g})")
        if not t["evidence_ok"]:
            reasons.append("evidence quote not found in document")
        if pd.isna(t["vendor_id"]) or pd.isna(t["item"]):
            reasons.append("vendor or item not recognised")
        if pd.isna(t["unit_price"]) and pd.isna(t["rebate_pct"]):
            reasons.append("no price or rebate")
        if pd.isna(t["effective_from"]):
            reasons.append("no start date")
        return "; ".join(reasons)
    return pd.Series([why(t) for _, t in extracted.iterrows()], index=extracted.index, dtype=str)


def review_queue(extracted: pd.DataFrame) -> pd.DataFrame:
    """Terms not to act on without a human check, with the reasons."""
    why = _review_reasons(extracted)
    return extracted.assign(why=why)[why != ""].reset_index(drop=True)


def trusted_terms(extracted: pd.DataFrame) -> pd.DataFrame:
    """Extracted terms that pass every review check: the only ones used for claims."""
    return extracted[_review_reasons(extracted) == ""].reset_index(drop=True)
