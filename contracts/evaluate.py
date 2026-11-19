"""Score extracted terms against the answer key, and pick out terms a human should check."""
import pandas as pd

from contracts.terms import to_ids
from leakage.clean import AUTO_MATCH

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


def field_accuracy(extracted: pd.DataFrame, truth: pd.DataFrame,
                   found_only: bool = False) -> pd.DataFrame:
    """Share of true terms found, then, per field, the share of true values extracted exactly.

    Each field is scored only over true terms that have a value for it (a contract price has
    no rebate fields), so empty fields can't pad the score. wrongly_filled counts values the
    model added where the truth is empty. found_only scores just the terms that were found
    (missed terms already show in the 'found' row)."""
    pairs, _ = _match(extracted, truth)
    found = sum(e is not None for _, e in pairs)
    rows = [{"field": "found", "correct": found, "total": len(pairs), "wrongly_filled": 0}]
    for f in FIELDS:
        scored = [(t, e) for t, e in pairs if pd.notna(t[f]) and (e is not None or not found_only)]
        rows.append({"field": f, "total": len(scored),
                     "correct": sum(e is not None and _same(e[f], t[f]) for t, e in scored),
                     "wrongly_filled": sum(e is not None and pd.isna(t[f]) and pd.notna(e[f])
                                           for t, e in pairs)})
    out = pd.DataFrame(rows, columns=["field", "correct", "total", "wrongly_filled"])
    out["accuracy"] = [round(c / t, 3) if t else None for c, t in zip(out["correct"], out["total"])]
    return out


def extra_terms(extracted: pd.DataFrame, truth: pd.DataFrame) -> int:
    """Extracted terms that match no true term: hallucinations, or traps the model fell for."""
    return len(_match(extracted, truth)[1])


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
        elif t["vendor_match_score"] < AUTO_MATCH:
            reasons.append(f"vendor name matched loosely ({t['vendor_match_score']:g})")
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
